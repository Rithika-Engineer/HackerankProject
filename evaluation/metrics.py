"""
Evaluation Metrics
"""
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    # Ensure they align
    if "user_id" not in y_true.columns or "user_id" not in y_pred.columns:
        return {"error": "Missing user_id"}
        
    merged = pd.merge(y_true, y_pred, on="user_id", suffixes=('_true', '_pred'))
    
    if merged.empty:
        return {"error": "No overlapping user_ids"}

    metrics = {}
    
    # 1. Exact Match columns
    for col in ["claim_status", "issue_type", "object_part", "severity", "evidence_standard_met", "valid_image"]:
        if f"{col}_true" in merged.columns and f"{col}_pred" in merged.columns:
            true_vals = merged[f"{col}_true"].astype(str).str.lower()
            pred_vals = merged[f"{col}_pred"].astype(str).str.lower()
            acc = accuracy_score(true_vals, pred_vals)
            metrics[f"{col}_accuracy"] = round(acc, 4)
            
    # claim_status precision/recall
    if "claim_status_true" in merged.columns and "claim_status_pred" in merged.columns:
        true_vals = merged["claim_status_true"].astype(str).str.lower()
        pred_vals = merged["claim_status_pred"].astype(str).str.lower()
        
        # calculate macro metrics
        metrics["claim_status_macro_precision"] = round(precision_score(true_vals, pred_vals, average="macro", zero_division=0), 4)
        metrics["claim_status_macro_recall"] = round(recall_score(true_vals, pred_vals, average="macro", zero_division=0), 4)
        metrics["claim_status_macro_f1"] = round(f1_score(true_vals, pred_vals, average="macro", zero_division=0), 4)

    return metrics
