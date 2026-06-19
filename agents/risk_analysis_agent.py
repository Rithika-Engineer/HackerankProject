"""
Risk Analysis Agent
Aggregates risk flags from user history, image quality, and authenticity signals.
This agent is rule-based (no LLM call needed) for efficiency.
"""

import logging

logger = logging.getLogger(__name__)


def analyze_risk(
    user_history: dict,
    image_analyses: dict,
    claim_object: str,
    claimed_object_part: str,
) -> list[str]:
    """
    Aggregate risk flags from multiple sources.

    Risk flag sources:
    1. User history flags (from user_history.csv)
    2. Image quality flags (from image analysis)
    3. Image authenticity flags (from image analysis)
    4. Object/part mismatch flags (comparison)

    Args:
        user_history: User history dict with 'history_flags', 'rejected_claim', etc.
        image_analyses: Output from image_analysis_agent
        claim_object: Expected object type
        claimed_object_part: Expected object part

    Returns:
        List of risk flag strings, or ['none'] if no flags
    """
    flags = set()

    # 1. User history flags
    history_flags_str = user_history.get("history_flags", "none")
    if history_flags_str and history_flags_str != "none":
        for flag in history_flags_str.split(";"):
            flag = flag.strip()
            if flag:
                flags.add(flag)

    # Check if user has high rejection rate
    rejected = int(user_history.get("rejected_claim", 0))
    total = int(user_history.get("past_claim_count", 0))
    recent_claims = int(user_history.get("last_90_days_claim_count", 0))

    if total > 0 and rejected / total >= 0.4:
        flags.add("user_history_risk")
    if recent_claims >= 5:
        flags.add("user_history_risk")

    # 2. Image quality and authenticity flags from analyses
    for analysis in image_analyses.get("image_analyses", []):
        quality_flags = analysis.get("quality_flags", [])
        for qf in quality_flags:
            if qf in [
                "blurry_image",
                "cropped_or_obstructed",
                "low_light_or_glare",
                "wrong_angle",
            ]:
                flags.add(qf)

        authenticity_flags = analysis.get("authenticity_flags", [])
        for af in authenticity_flags:
            if af in [
                "non_original_image",
                "possible_manipulation",
                "text_instruction_present",
            ]:
                flags.add(af)

    # 3. Object mismatch detection
    for analysis in image_analyses.get("image_analyses", []):
        detected_object = analysis.get("detected_object", "").lower()
        if detected_object and detected_object != claim_object and detected_object != "unknown":
            flags.add("wrong_object")

        detected_part = analysis.get("detected_object_part", "").lower()
        if (
            detected_part
            and claimed_object_part
            and detected_part != claimed_object_part
            and detected_part != "unknown"
            and claimed_object_part != "unknown"
        ):
            # Only flag if the detected part is clearly different
            # and the claimed part is not visible in any image
            all_detected_parts = [
                a.get("detected_object_part", "").lower()
                for a in image_analyses.get("image_analyses", [])
            ]
            if claimed_object_part not in all_detected_parts:
                flags.add("wrong_object_part")

    # 4. Claim mismatch detection (damage type vs visible)
    for analysis in image_analyses.get("image_analyses", []):
        detected_issue = analysis.get("detected_issue_type", "").lower()
        if detected_issue == "none":
            flags.add("damage_not_visible")

    # 5. If user_history_risk or non_original_image or text_instruction present,
    # add manual_review_required
    high_risk_flags = {
        "user_history_risk",
        "non_original_image",
        "possible_manipulation",
        "text_instruction_present",
        "wrong_object",
        "claim_mismatch",
    }
    if flags & high_risk_flags:
        flags.add("manual_review_required")

    if not flags:
        return ["none"]

    # Sort for consistency
    return sorted(list(flags))
