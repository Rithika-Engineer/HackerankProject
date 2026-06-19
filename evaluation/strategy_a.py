"""
Evaluation Strategy A: Single Vision Prompt
Processes claims in a single LLM call per claim.
"""
import os
import sys
import time
import json
import logging
from pathlib import Path

CODE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(CODE_DIR))

from google import genai
from google.genai import types
import pandas as pd
from services.image_loader import load_images_for_claim

logger = logging.getLogger(__name__)

SINGLE_PROMPT = """
You are a claims adjudicator verifying a damage claim.
Claim Object: {claim_object}
User Claim: {user_claim}
User History Context: {history_context}
Evidence Requirements: {evidence_requirements}

Please review the attached images carefully.
Output a JSON object with the following fields (all lowercase string values matching allowed sets):
- evidence_standard_met (true/false)
- evidence_standard_met_reason
- risk_flags (semicolon separated)
- issue_type (dent, scratch, crack, etc)
- object_part 
- claim_status (supported, contradicted, not_enough_information)
- claim_status_justification
- supporting_image_ids (semicolon separated, or none)
- valid_image (true/false)
- severity (none, low, medium, high, unknown)
"""

def evaluate_strategy_a(df: pd.DataFrame, history_service, evidence_reqs, client, model_name, dataset_dir) -> dict:
    results = []
    stats = {"llm_calls": 0, "errors": 0}
    start_time = time.time()
    
    for idx, row in df.iterrows():
        try:
            images = load_images_for_claim(row["image_paths"], dataset_dir)
            
            # Format prompt
            history = history_service.get_user_history(row["user_id"])
            history_ctx = f"Past claims: {history.get('past_claim_count')}, rejected: {history.get('rejected_claim')}, flags: {history.get('history_flags')}"
            
            prompt_text = SINGLE_PROMPT.format(
                claim_object=row["claim_object"],
                user_claim=row["user_claim"],
                history_context=history_ctx,
                evidence_requirements=json.dumps([req for req in evidence_reqs if req['claim_object'] in ('all', row["claim_object"])])
            )
            
            content_parts = []
            valid_images = [img for img in images if img.get("valid")]
            for img in valid_images:
                import base64
                image_bytes = base64.b64decode(img["base64_data"])
                content_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=img["mime_type"]))
                content_parts.append(types.Part.from_text(text=f"[Above image is: {img['image_id']}]"))
            content_parts.append(types.Part.from_text(text=prompt_text))
            
            response = client.models.generate_content(
                model=model_name,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            )
            stats["llm_calls"] += 1
            res = json.loads(response.text)
            res["user_id"] = row["user_id"]
            results.append(res)
            
            time.sleep(1.0) # rate limit
            
        except Exception as e:
            logger.error(f"Strategy A Error on row {idx}: {e}")
            stats["errors"] += 1
            
    elapsed = time.time() - start_time
    stats["total_time_seconds"] = elapsed
    
    return {"predictions": pd.DataFrame(results), "stats": stats}
