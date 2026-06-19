"""
Main Orchestrator for AI Claim Verification System
Multi-Agent Pipeline: Claim Extraction → Image Analysis → Evidence Validation
                      → Risk Analysis → Decision → Explanation → Output

Usage:
    python main.py --mode test     # Process claims.csv → output.csv
    python main.py --mode sample   # Process sample_claims.csv for evaluation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai

# Setup paths
CODE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

sys.path.insert(0, str(CODE_DIR))

from agents.claim_extractor_agent import extract_claim
from agents.image_analysis_agent import analyze_images
from agents.evidence_validation_agent import validate_evidence
from agents.risk_analysis_agent import analyze_risk
from agents.decision_agent import make_decision
from agents.explanation_agent import generate_explanation
from services.csv_loader import (
    load_claims,
    load_sample_claims,
    load_user_history,
    load_evidence_requirements,
)
from services.image_loader import load_images_for_claim
from services.history_service import UserHistoryService
from services.output_writer import write_output, validate_output

# Load .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(CODE_DIR / "pipeline.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

# Configuration
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
RATE_LIMIT_DELAY = float(os.environ.get("RATE_LIMIT_DELAY", "1.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Required output columns in order
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


def init_client() -> genai.Client:
    """Initialize the Gemini API client."""
    if not API_KEY:
        logger.error(
            "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )
        sys.exit(1)
    client = genai.Client(api_key=API_KEY)
    logger.info(f"Gemini client initialized with model: {MODEL_NAME}")
    return client


def process_single_claim(
    row: pd.Series,
    history_service: UserHistoryService,
    evidence_requirements: list[dict],
    client: genai.Client,
    stats: dict,
) -> dict:
    """
    Process a single claim through the full multi-agent pipeline.

    Pipeline:
    1. Claim Extraction (text-only LLM call)
    2. Image Analysis (vision LLM call)
    3. Evidence Validation (text LLM call)
    4. Risk Analysis (rule-based, no LLM)
    5. Decision (text LLM call)
    6. Explanation (rule-based refinement, no LLM)

    Args:
        row: DataFrame row with claim data
        history_service: User history lookup service
        evidence_requirements: List of evidence requirement dicts
        client: Gemini client
        stats: Mutable stats dict for tracking

    Returns:
        dict with all output columns
    """
    user_id = row["user_id"]
    image_paths_str = row["image_paths"]
    user_claim = row["user_claim"]
    claim_object = row["claim_object"]

    logger.info(f"Processing claim for {user_id} ({claim_object})")

    # ── Agent 1: Claim Extraction ──
    logger.info(f"  [Agent 1] Extracting claim...")
    claim_extraction = extract_claim(user_claim, claim_object, client, MODEL_NAME)
    stats["llm_calls"] += 1
    time.sleep(RATE_LIMIT_DELAY)

    # ── Agent 2: Image Analysis ──
    logger.info(f"  [Agent 2] Analyzing images...")
    images = load_images_for_claim(image_paths_str, str(DATASET_DIR))
    stats["images_processed"] += len(images)

    image_analyses = analyze_images(images, claim_object, client, MODEL_NAME)
    stats["llm_calls"] += 1
    time.sleep(RATE_LIMIT_DELAY)

    # ── Agent 3: Evidence Validation ──
    logger.info(f"  [Agent 3] Validating evidence...")
    evidence_validation = validate_evidence(
        claim_object=claim_object,
        claimed_issue_type=claim_extraction.get("claimed_issue_type", "unknown"),
        claimed_object_part=claim_extraction.get("claimed_object_part", "unknown"),
        image_analyses=image_analyses,
        evidence_requirements=evidence_requirements,
        client=client,
        model_name=MODEL_NAME,
    )
    stats["llm_calls"] += 1
    time.sleep(RATE_LIMIT_DELAY)

    # ── Agent 4: Risk Analysis (rule-based) ──
    logger.info(f"  [Agent 4] Analyzing risk...")
    user_history = history_service.get_user_history(user_id)
    risk_flags = analyze_risk(
        user_history=user_history,
        image_analyses=image_analyses,
        claim_object=claim_object,
        claimed_object_part=claim_extraction.get("claimed_object_part", "unknown"),
    )

    # ── Agent 5: Decision ──
    logger.info(f"  [Agent 5] Making decision...")
    decision = make_decision(
        claim_object=claim_object,
        claim_extraction=claim_extraction,
        image_analyses=image_analyses,
        evidence_validation=evidence_validation,
        risk_flags=risk_flags,
        user_history=user_history,
        client=client,
        model_name=MODEL_NAME,
    )
    stats["llm_calls"] += 1
    time.sleep(RATE_LIMIT_DELAY)

    # ── Agent 6: Explanation ──
    logger.info(f"  [Agent 6] Generating explanation...")
    explanation = generate_explanation(
        decision=decision,
        claim_extraction=claim_extraction,
        image_analyses=image_analyses,
        evidence_validation=evidence_validation,
    )

    # ── Assemble final output ──
    supporting_ids = decision.get("supporting_image_ids", [])
    if isinstance(supporting_ids, list):
        supporting_ids_str = ";".join(supporting_ids) if supporting_ids else "none"
    else:
        supporting_ids_str = supporting_ids if supporting_ids else "none"

    final_risk_flags = decision.get("risk_flags", risk_flags)
    if isinstance(final_risk_flags, list):
        risk_flags_str = ";".join(final_risk_flags) if final_risk_flags else "none"
    else:
        risk_flags_str = final_risk_flags if final_risk_flags else "none"

    # Convert evidence_standard_met and valid_image to lowercase string booleans
    evidence_met = evidence_validation.get("evidence_standard_met", False)
    valid_img = evidence_validation.get("valid_image", True)

    result = {
        "user_id": user_id,
        "image_paths": image_paths_str,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": str(evidence_met).lower(),
        "evidence_standard_met_reason": explanation.get(
            "evidence_standard_met_reason",
            evidence_validation.get("evidence_standard_met_reason", ""),
        ),
        "risk_flags": risk_flags_str,
        "issue_type": decision.get("issue_type", "unknown"),
        "object_part": decision.get("object_part", "unknown"),
        "claim_status": decision.get("claim_status", "not_enough_information"),
        "claim_status_justification": explanation.get(
            "claim_status_justification",
            decision.get("claim_status_justification", ""),
        ),
        "supporting_image_ids": supporting_ids_str,
        "valid_image": str(valid_img).lower(),
        "severity": decision.get("severity", "unknown"),
    }

    logger.info(
        f"  → Result: status={result['claim_status']}, "
        f"issue={result['issue_type']}, part={result['object_part']}, "
        f"severity={result['severity']}"
    )

    return result


def run_pipeline(mode: str = "test") -> pd.DataFrame:
    """
    Run the full claim verification pipeline.

    Args:
        mode: 'test' to process claims.csv, 'sample' to process sample_claims.csv

    Returns:
        DataFrame with all output columns
    """
    start_time = time.time()

    # Initialize
    client = init_client()

    # Load data
    logger.info("Loading datasets...")
    if mode == "sample":
        claims_df = load_sample_claims(str(DATASET_DIR / "sample_claims.csv"))
        logger.info(f"Loaded {len(claims_df)} sample claims")
    else:
        claims_df = load_claims(str(DATASET_DIR / "claims.csv"))
        logger.info(f"Loaded {len(claims_df)} test claims")

    history_df = load_user_history(str(DATASET_DIR / "user_history.csv"))
    history_service = UserHistoryService(history_df)
    logger.info(f"Loaded {len(history_df)} user history records")

    evidence_reqs_df = load_evidence_requirements(
        str(DATASET_DIR / "evidence_requirements.csv")
    )
    evidence_requirements = evidence_reqs_df.to_dict("records")
    logger.info(f"Loaded {len(evidence_requirements)} evidence requirements")

    # Process each claim
    results = []
    stats = {"llm_calls": 0, "images_processed": 0, "errors": 0}

    for idx, row in claims_df.iterrows():
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Claim {idx + 1}/{len(claims_df)}")
            result = process_single_claim(
                row, history_service, evidence_requirements, client, stats
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing claim {idx + 1}: {e}", exc_info=True)
            stats["errors"] += 1
            # Generate fallback result
            results.append(
                {
                    "user_id": row["user_id"],
                    "image_paths": row["image_paths"],
                    "user_claim": row["user_claim"],
                    "claim_object": row["claim_object"],
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": "Processing error occurred",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Claim could not be processed due to a system error.",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown",
                }
            )

    # Build output DataFrame
    output_df = pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete!")
    logger.info(f"  Claims processed: {len(results)}")
    logger.info(f"  LLM calls: {stats['llm_calls']}")
    logger.info(f"  Images processed: {stats['images_processed']}")
    logger.info(f"  Errors: {stats['errors']}")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Avg time per claim: {elapsed/len(results):.1f}s")

    # Save stats for evaluation
    stats_path = str(CODE_DIR / "pipeline_stats.json")
    stats["total_time_seconds"] = elapsed
    stats["claims_processed"] = len(results)
    stats["mode"] = mode
    stats["model"] = MODEL_NAME
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return output_df


def main():
    parser = argparse.ArgumentParser(
        description="AI Claim Verification System - Multi-Agent Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "sample"],
        default="test",
        help="'test' to process claims.csv (default), 'sample' for evaluation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: dataset/output.csv for test, code/sample_output.csv for sample)",
    )
    args = parser.parse_args()

    # Run pipeline
    output_df = run_pipeline(mode=args.mode)

    # Determine output path
    if args.output:
        output_path = args.output
    elif args.mode == "test":
        output_path = str(DATASET_DIR / "output.csv")
    else:
        output_path = str(CODE_DIR / "sample_output.csv")

    # Validate and write output
    validation_errors = validate_output(output_df)
    if validation_errors:
        logger.warning(f"Output validation issues:")
        for err in validation_errors:
            logger.warning(f"  - {err}")

    write_output(output_df, output_path)
    logger.info(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
