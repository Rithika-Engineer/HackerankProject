"""
Explanation Agent
Generates concise, image-grounded justifications for claim decisions.
This agent refines the justification from the decision agent if needed.
"""

import logging

logger = logging.getLogger(__name__)


def generate_explanation(
    decision: dict,
    claim_extraction: dict,
    image_analyses: dict,
    evidence_validation: dict,
) -> dict:
    """
    Refine and validate the explanation/justification from the decision agent.
    This is a lightweight post-processing step (no additional LLM call).

    Args:
        decision: Output from decision_agent
        claim_extraction: Output from claim_extractor_agent
        image_analyses: Output from image_analysis_agent
        evidence_validation: Output from evidence_validation_agent

    Returns:
        dict with:
            - claim_status_justification: str (refined)
            - evidence_standard_met_reason: str (refined)
    """
    justification = decision.get("claim_status_justification", "")
    evidence_reason = evidence_validation.get("evidence_standard_met_reason", "")

    # Ensure justification references image IDs when possible
    supporting_ids = decision.get("supporting_image_ids", [])
    claim_status = decision.get("claim_status", "")

    if not justification:
        justification = _generate_default_justification(
            claim_status, decision, claim_extraction, image_analyses
        )

    if not evidence_reason:
        evidence_reason = _generate_default_evidence_reason(
            evidence_validation, claim_extraction
        )

    # Validate justification length — keep concise (1-2 sentences)
    if len(justification) > 500:
        # Truncate to first two sentences
        sentences = justification.split(". ")
        justification = ". ".join(sentences[:2])
        if not justification.endswith("."):
            justification += "."

    return {
        "claim_status_justification": justification,
        "evidence_standard_met_reason": evidence_reason,
    }


def _generate_default_justification(
    claim_status: str,
    decision: dict,
    claim_extraction: dict,
    image_analyses: dict,
) -> str:
    """Generate a default justification when none is provided."""
    claimed_issue = claim_extraction.get("claimed_issue_type", "unknown")
    claimed_part = claim_extraction.get("claimed_object_part", "unknown")
    actual_issue = decision.get("issue_type", "unknown")
    actual_part = decision.get("object_part", "unknown")
    supporting_ids = decision.get("supporting_image_ids", [])

    if claim_status == "supported":
        img_ref = f" in {supporting_ids[0]}" if supporting_ids else ""
        return (
            f"The image{img_ref} shows {actual_issue} on the {actual_part}, "
            f"consistent with the user's claim."
        )
    elif claim_status == "contradicted":
        if actual_issue == "none":
            return (
                f"The {claimed_part} is visible in the image but shows no damage, "
                f"contradicting the user's {claimed_issue} claim."
            )
        else:
            return (
                f"The image shows {actual_issue} on the {actual_part}, which "
                f"does not match the claimed {claimed_issue} on {claimed_part}."
            )
    else:  # not_enough_information
        return (
            f"The submitted images do not clearly show the claimed {claimed_part}, "
            f"so the {claimed_issue} claim cannot be verified."
        )


def _generate_default_evidence_reason(
    evidence_validation: dict,
    claim_extraction: dict,
) -> str:
    """Generate a default evidence reason when none is provided."""
    met = evidence_validation.get("evidence_standard_met", False)
    claimed_part = claim_extraction.get("claimed_object_part", "unknown")

    if met:
        return f"The {claimed_part} is visible and can be evaluated from the submitted images."
    else:
        return f"The submitted images do not clearly show the {claimed_part} for evaluation."
