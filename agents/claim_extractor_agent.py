"""
Claim Extractor Agent
Extracts claimed damage details from user conversation text.
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
    """Load the claim extraction prompt template."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts",
        "claim_extraction_prompt.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_claim(
    user_claim: str,
    claim_object: str,
    client: genai.Client,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Extract claimed damage details from a user conversation.

    Args:
        user_claim: The full customer support conversation text.
        claim_object: The object type ('car', 'laptop', 'package').
        client: Initialized Gemini client.
        model_name: Model to use for extraction.

    Returns:
        dict with keys:
            - claimed_issue_type: str
            - claimed_object_part: str
            - damage_keywords: list[str]
            - claimed_severity_description: str
            - num_parts_claimed: int
    """
    prompt_template = _load_prompt_template()
    prompt = prompt_template.replace("{user_claim}", user_claim).replace(
        "{claim_object}", claim_object
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
        logger.info(
            f"Claim extraction result: issue={result.get('claimed_issue_type')}, "
            f"part={result.get('claimed_object_part')}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse claim extraction response: {e}")
        logger.debug(f"Raw response: {response.text}")
        return {
            "claimed_issue_type": "unknown",
            "claimed_object_part": "unknown",
            "damage_keywords": [],
            "claimed_severity_description": "unknown",
            "num_parts_claimed": 1,
        }
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return {
            "claimed_issue_type": "unknown",
            "claimed_object_part": "unknown",
            "damage_keywords": [],
            "claimed_severity_description": "unknown",
            "num_parts_claimed": 1,
        }
