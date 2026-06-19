"""
Configuration file for the AI Claim Verification System.
"""

import os
from pathlib import Path

# Paths
CODE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"

# Load environment variables
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# API Configuration
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

# Resiliency
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", "2.0"))
REQUESTS_PER_MINUTE = int(os.environ.get("REQUESTS_PER_MINUTE", "15"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))

# Output Schema
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

ALLOWED_VALUES = {
    "claim_status": ["supported", "contradicted", "not_enough_information"],
    "issue_type": [
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown"
    ],
    "object_part_car": [
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown"
    ],
    "object_part_laptop": [
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown"
    ],
    "object_part_package": [
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown"
    ],
    "risk_flags": [
        "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
        "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
        "claim_mismatch", "possible_manipulation", "non_original_image",
        "text_instruction_present", "user_history_risk", "manual_review_required"
    ],
    "severity": ["none", "low", "medium", "high", "unknown"]
}
