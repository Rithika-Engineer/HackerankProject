"""
CSV loading utilities.
"""

import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def load_claims(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        # image_paths should be handled as string directly for easier formatting, but we might need lists internally
        # We will keep it as string in df, but functions parsing it can split by ';'
        return df
    except Exception as e:
        logger.error(f"Failed to load claims from {filepath}: {e}")
        raise

def load_sample_claims(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load sample claims from {filepath}: {e}")
        raise

def load_user_history(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load user history from {filepath}: {e}")
        raise

def load_evidence_requirements(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        logger.error(f"Failed to load evidence requirements from {filepath}: {e}")
        raise
