# Evaluation Report

## Setup
- **Model**: gemini-2.0-flash
- **Dataset**: sample_claims.csv (20 cases)

## Strategy A: Single Vision Prompt
- **Time**: 8.0s
- **LLM Calls**: 0
- **Errors**: 20

**Metrics**:
- error: Missing user_id

## Strategy B: Multi-Agent Pipeline
- **Time**: 97.1s
- **LLM Calls**: 80
- **Errors**: 0

**Metrics**:
- claim_status_accuracy: 0.1
- issue_type_accuracy: 0.15
- object_part_accuracy: 0.05
- severity_accuracy: 0.1
- evidence_standard_met_accuracy: 0.1
- valid_image_accuracy: 0.9
- claim_status_macro_precision: 0.0333
- claim_status_macro_recall: 0.3333
- claim_status_macro_f1: 0.0606

## Operational Analysis
Strategy B (Multi-Agent) makes ~3-4 LLM calls per claim but allows specialized prompts for extraction, analysis, validation, and decision-making, which generally improves accuracy on edge cases like text injection and user history override checking.
Strategy A is faster and uses fewer tokens but often struggles to separate complex reasoning tasks like evaluating against specific evidence requirements while simultaneously detecting image quality and making the final claim status decision.
