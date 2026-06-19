"""
Output writer.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

from config import OUTPUT_COLUMNS, ALLOWED_VALUES

def write_output(df: pd.DataFrame, output_path: str):
    """Write output CSV with exact required schema."""
    # Ensure all columns exist, add with empty/default if missing
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            logger.warning(f"Missing column {col} in output, adding with empty string")
            df[col] = ""
            
    # Reorder columns
    df = df[OUTPUT_COLUMNS]
    df.to_csv(output_path, index=False, quoting=1) # Quote non-numeric

def validate_output(df: pd.DataFrame) -> list[str]:
    """Check all required columns exist and values are valid."""
    errors = []
    
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
            
    if "claim_status" in df.columns:
        invalid_statuses = df[~df["claim_status"].isin(ALLOWED_VALUES["claim_status"])]
        if not invalid_statuses.empty:
            errors.append(f"Invalid claim_status found: {invalid_statuses['claim_status'].unique()}")
            
    return errors
