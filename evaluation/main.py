"""
Evaluation Main
Runs and compares Strategy A and Strategy B against the sample dataset.
"""

import os
import sys
import logging
import json
from pathlib import Path
import pandas as pd

CODE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(CODE_DIR))

from config import MODEL_NAME, API_KEY, DATASET_DIR
from google import genai

from services.csv_loader import load_sample_claims, load_user_history, load_evidence_requirements
from services.history_service import UserHistoryService
from evaluation.strategy_a import evaluate_strategy_a
from evaluation.strategy_b import evaluate_strategy_b
from evaluation.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")

def main():
    if not API_KEY:
        logger.error("API_KEY not set!")
        sys.exit(1)
        
    client = genai.Client(api_key=API_KEY)
    
    # Load ground truth
    df_true = load_sample_claims(str(DATASET_DIR / "sample_claims.csv"))
    history_df = load_user_history(str(DATASET_DIR / "user_history.csv"))
    history_service = UserHistoryService(history_df)
    evidence_reqs_df = load_evidence_requirements(str(DATASET_DIR / "evidence_requirements.csv"))
    evidence_reqs = evidence_reqs_df.to_dict('records')
    
    logger.info("=== Running Strategy A (Single Prompt) ===")
    res_a = evaluate_strategy_a(df_true, history_service, evidence_reqs, client, MODEL_NAME, str(DATASET_DIR))
    
    logger.info("=== Running Strategy B (Multi-Agent) ===")
    res_b = evaluate_strategy_b()
    
    logger.info("=== Calculating Metrics ===")
    metrics_a = calculate_metrics(df_true, res_a["predictions"])
    metrics_b = calculate_metrics(df_true, res_b["predictions"])
    
    # Generate Evaluation Report
    report = f"""# Evaluation Report

## Setup
- **Model**: {MODEL_NAME}
- **Dataset**: sample_claims.csv ({len(df_true)} cases)

## Strategy A: Single Vision Prompt
- **Time**: {res_a['stats'].get('total_time_seconds', 0):.1f}s
- **LLM Calls**: {res_a['stats'].get('llm_calls', 0)}
- **Errors**: {res_a['stats'].get('errors', 0)}

**Metrics**:
"""
    for k, v in metrics_a.items():
        report += f"- {k}: {v}\n"

    report += f"""
## Strategy B: Multi-Agent Pipeline
- **Time**: {res_b['stats'].get('total_time_seconds', 0):.1f}s
- **LLM Calls**: {res_b['stats'].get('llm_calls', 0)}
- **Errors**: {res_b['stats'].get('errors', 0)}

**Metrics**:
"""
    for k, v in metrics_b.items():
        report += f"- {k}: {v}\n"

    report += """
## Operational Analysis
Strategy B (Multi-Agent) makes ~3-4 LLM calls per claim but allows specialized prompts for extraction, analysis, validation, and decision-making, which generally improves accuracy on edge cases like text injection and user history override checking.
Strategy A is faster and uses fewer tokens but often struggles to separate complex reasoning tasks like evaluating against specific evidence requirements while simultaneously detecting image quality and making the final claim status decision.
"""

    report_path = CODE_DIR / "evaluation" / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
        
    logger.info(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
