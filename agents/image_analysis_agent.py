"""
Image Analysis Agent
Analyzes all submitted images for a claim using Gemini Vision.
Detects damage type, object part, severity, quality issues, and authenticity flags.
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
    """Load the image analysis prompt template."""
    prompt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "prompts",
        "image_analysis_prompt.txt",
    )
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def analyze_images(
    images: list[dict],
    claim_object: str,
    client: genai.Client,
    model_name: str = "gemini-2.0-flash",
) -> dict:
    """
    Analyze all images for a claim using Gemini Vision.

    Args:
        images: List of image dicts from image_loader, each with:
            - image_id: str (e.g., 'img_1')
            - base64_data: str (base64-encoded image bytes)
            - mime_type: str (e.g., 'image/jpeg')
            - valid: bool
            - path: str
        claim_object: The object type ('car', 'laptop', 'package').
        client: Initialized Gemini client.
        model_name: Model to use for analysis.

    Returns:
        dict with:
            - image_analyses: list of per-image analysis dicts
            - overall_quality_flags: list[str]
            - overall_authenticity_flags: list[str]
    """
    # Filter to valid images only
    valid_images = [img for img in images if img.get("valid", False)]

    if not valid_images:
        logger.warning("No valid images to analyze")
        return {
            "image_analyses": [],
            "overall_quality_flags": ["damage_not_visible"],
            "overall_authenticity_flags": [],
        }

    prompt_template = _load_prompt_template()
    prompt_text = prompt_template.replace("{claim_object}", claim_object).replace(
        "{num_images}", str(len(valid_images))
    )

    # Build multimodal content: text prompt + all images
    content_parts = []

    # Add each image as an inline part
    for img in valid_images:
        import base64

        image_bytes = base64.b64decode(img["base64_data"])
        content_parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=img["mime_type"],
            )
        )
        content_parts.append(
            types.Part.from_text(text=f"[Above image is: {img['image_id']}]")
        )

    # Add the analysis prompt at the end
    content_parts.append(types.Part.from_text(text=prompt_text))

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text)

        # Ensure we have the expected structure
        if "image_analyses" not in result:
            # If the model returns a flat list, wrap it
            if isinstance(result, list):
                result = {
                    "image_analyses": result,
                    "overall_quality_flags": [],
                    "overall_authenticity_flags": [],
                }
            else:
                result = {
                    "image_analyses": [result],
                    "overall_quality_flags": [],
                    "overall_authenticity_flags": [],
                }

        # Aggregate quality and authenticity flags across all images
        all_quality_flags = set()
        all_authenticity_flags = set()
        for analysis in result.get("image_analyses", []):
            all_quality_flags.update(analysis.get("quality_flags", []))
            all_authenticity_flags.update(analysis.get("authenticity_flags", []))

        result["overall_quality_flags"] = list(all_quality_flags)
        result["overall_authenticity_flags"] = list(all_authenticity_flags)

        logger.info(
            f"Image analysis complete: {len(result['image_analyses'])} images analyzed"
        )
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse image analysis response: {e}")
        logger.debug(f"Raw response: {response.text}")
        return {
            "image_analyses": [
                {
                    "image_id": img["image_id"],
                    "detected_object": "unknown",
                    "detected_issue_type": "unknown",
                    "detected_object_part": "unknown",
                    "detected_severity": "unknown",
                    "quality_flags": [],
                    "authenticity_flags": [],
                    "description": "Analysis failed",
                }
                for img in valid_images
            ],
            "overall_quality_flags": [],
            "overall_authenticity_flags": [],
        }
    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return {
            "image_analyses": [],
            "overall_quality_flags": [],
            "overall_authenticity_flags": [],
        }
