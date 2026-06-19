"""
Evidence Validation Agent
Validates whether submitted image evidence meets minimum requirements.
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
    """Load the evidence validation prompt template."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts",
        "evidence_validation_prompt.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def _get_applicable_requirements(
    claim_object: str,
    claimed_issue_type: str,
    evidence_requirements: list[dict],
) -> list[dict]:
    """
    Filter evidence requirements to those applicable to this claim.

    Args:
        claim_object: 'car', 'laptop', or 'package'
        claimed_issue_type: The issue type from claim extraction
        evidence_requirements: Full list of requirement dicts

    Returns:
        List of applicable requirement dicts
    """
    applicable = []
    for req in evidence_requirements:
        obj = req.get("claim_object", "")
        if obj == "all" or obj == claim_object:
            applicable.append(req)
    return applicable


def validate_evidence(
    claim_object: str,
    claimed_issue_type: str,
    claimed_object_part: str,
    image_analyses: dict,
    evidence_requirements: list[dict],
    client: genai.Client,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Validate whether submitted images meet evidence standards.

    Args:
        claim_object: 'car', 'laptop', or 'package'
        claimed_issue_type: Extracted issue type from claim
        claimed_object_part: Extracted object part from claim
        image_analyses: Output from image_analysis_agent
        evidence_requirements: List of requirement dicts from CSV
        client: Initialized Gemini client
        model_name: Model name

    Returns:
        dict with:
            - evidence_standard_met: bool
            - evidence_standard_met_reason: str
            - valid_image: bool
            - applicable_requirements: list[str]
    """
    applicable_reqs = _get_applicable_requirements(
        claim_object, claimed_issue_type, evidence_requirements
    )

    prompt_template = _load_prompt_template()
    prompt = (
        prompt_template.replace("{claim_object}", claim_object)
        .replace("{claimed_issue_type}", claimed_issue_type)
        .replace("{claimed_object_part}", claimed_object_part)
        .replace("{image_analyses_json}", json.dumps(image_analyses, indent=2))
        .replace("{evidence_requirements_json}", json.dumps(applicable_reqs, indent=2))
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

        # Ensure boolean types
        result["evidence_standard_met"] = bool(result.get("evidence_standard_met", False))
        result["valid_image"] = bool(result.get("valid_image", True))

        logger.info(
            f"Evidence validation: met={result['evidence_standard_met']}, "
            f"valid={result['valid_image']}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evidence validation response: {e}")
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Evidence validation failed due to processing error",
            "valid_image": True,
            "applicable_requirements": [],
        }
    except Exception as e:
        logger.error(f"Evidence validation failed: {e}")
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Evidence validation encountered an error",
            "valid_image": True,
            "applicable_requirements": [],
        }
