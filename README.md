# AI Claim Verification System

A multi-agent system to verify damage claims.

## Setup

1. Requirements: Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

## Running

To run on the test dataset (`claims.csv`) and produce `output.csv`:
```bash
python main.py --mode test
```

To run on the sample dataset for development:
```bash
python main.py --mode sample
```

## Evaluation
```bash
python evaluation/main.py
```
