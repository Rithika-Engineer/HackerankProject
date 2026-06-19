"""
Decision Agent
Makes the final claim status decision by comparing claim extraction
against image analysis results, evidence validation, and risk context.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _load_prompt_template() -> str:
    """Load the decision prompt template."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts",
        "decision_prompt.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def make_decision(
    claim_object: str,
    claim_extraction: dict,
    image_analyses: dict,
    evidence_validation: dict,
    risk_flags: list[str],
    user_history: dict,
    client: genai.Client,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Make the final claim verification decision.

    Args:
        claim_object: 'car', 'laptop', or 'package'
        claim_extraction: Output from claim_extractor_agent
        image_analyses: Output from image_analysis_agent
        evidence_validation: Output from evidence_validation_agent
        risk_flags: Output from risk_analysis_agent
        user_history: User history dict
        client: Initialized Gemini client
        model_name: Model name

    Returns:
        dict with:
            - claim_status: str ('supported', 'contradicted', 'not_enough_information')
            - issue_type: str (what's actually visible)
            - object_part: str (what's actually shown)
            - severity: str
            - claim_status_justification: str
            - supporting_image_ids: list[str]
            - risk_flags: list[str] (final combined)
    """
    claimed_issue_type = claim_extraction.get("claimed_issue_type", "unknown")
    claimed_object_part = claim_extraction.get("claimed_object_part", "unknown")

    prompt_template = _load_prompt_template()
    prompt = (
        prompt_template.replace("{claim_object}", claim_object)
        .replace("{claimed_issue_type}", claimed_issue_type)
        .replace("{claimed_object_part}", claimed_object_part)
        .replace("{image_analyses_json}", json.dumps(image_analyses, indent=2))
        .replace("{evidence_validation_json}", json.dumps(evidence_validation, indent=2))
        .replace("{risk_flags}", ";".join(risk_flags))
        .replace(
            "{user_history_json}",
            json.dumps(
                {
                    "user_id": user_history.get("user_id", "unknown"),
                    "past_claim_count": user_history.get("past_claim_count", 0),
                    "rejected_claim": user_history.get("rejected_claim", 0),
                    "history_flags": user_history.get("history_flags", "none"),
                    "history_summary": user_history.get("history_summary", ""),
                },
                indent=2,
            ),
        )
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)

        # Ensure supporting_image_ids is a list
        sids = result.get("supporting_image_ids", [])
        if isinstance(sids, str):
            if sids.lower() == "none":
                result["supporting_image_ids"] = []
            else:
                result["supporting_image_ids"] = [s.strip() for s in sids.split(";")]

        # Merge risk_flags from decision with existing ones
        decision_flags = result.get("risk_flags", [])
        if isinstance(decision_flags, str):
            if decision_flags.lower() == "none":
                decision_flags = []
            else:
                decision_flags = [f.strip() for f in decision_flags.split(";")]

        # Combine all risk flags
        combined_flags = set(risk_flags) | set(decision_flags)
        combined_flags.discard("none")
        result["risk_flags"] = sorted(list(combined_flags)) if combined_flags else ["none"]

        # Validate claim_status
        valid_statuses = {"supported", "contradicted", "not_enough_information"}
        if result.get("claim_status", "") not in valid_statuses:
            logger.warning(
                f"Invalid claim_status: {result.get('claim_status')}, defaulting to not_enough_information"
            )
            result["claim_status"] = "not_enough_information"

        logger.info(
            f"Decision: status={result['claim_status']}, "
            f"issue={result.get('issue_type')}, "
            f"part={result.get('object_part')}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse decision response: {e}")
        logger.debug(f"Raw response: {response.text}")
        return _fallback_decision(
            claimed_issue_type, claimed_object_part, risk_flags
        )
    except Exception as e:
        logger.error(f"Decision agent failed: {e}")
        return _fallback_decision(
            claimed_issue_type, claimed_object_part, risk_flags
        )


def _fallback_decision(
    claimed_issue_type: str,
    claimed_object_part: str,
    risk_flags: list[str],
) -> dict:
    """Generate a safe fallback decision when the LLM fails."""
    return {
        "claim_status": "not_enough_information",
        "issue_type": "unknown",
        "object_part": claimed_object_part,
        "severity": "unknown",
        "claim_status_justification": "Unable to process the claim due to a system error. Manual review required.",
        "supporting_image_ids": [],
        "risk_flags": risk_flags if risk_flags != ["none"] else ["manual_review_required"],
    }
