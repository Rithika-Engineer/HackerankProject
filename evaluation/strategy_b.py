"""
Evaluation Strategy B: Multi-Agent Pipeline
Calls the main.py pipeline.
"""

import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(CODE_DIR))

from main import run_pipeline

def evaluate_strategy_b() -> dict:
    """Wrapper to run main pipeline in sample mode and return its output dataframe."""
    import json
    
    # Run the main pipeline on sample data
    # It generates pipeline_stats.json locally
    df = run_pipeline(mode="sample")
    
    stats = {}
    stats_file = CODE_DIR / "pipeline_stats.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            
    return {"predictions": df, "stats": stats}
