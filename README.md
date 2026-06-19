# Multi-Modal Evidence Review System (AI Claim Verification)

This repository contains a production-grade, multi-agent AI system designed to verify damage claims for cars, laptops, and packages. The system leverages the `gemini-2.0-flash` model to analyze text conversations and multiple visual evidence images, cross-referencing them against user history and strict evidence requirements.

## System Architecture

To handle the complex, multi-modal reasoning required for this task, the system is designed as a **6-Agent Sequential Pipeline**. This prevents the AI from being overwhelmed by a single massive prompt and ensures high accuracy and robustness against edge cases (like text injection attacks).

### The 6-Agent Pipeline
1. **Claim Extractor Agent (LLM - Text)**: Parses the customer support transcript to extract the claimed issue, object part, and severity. Handles multi-language inputs and confusing narratives to find the *final* claimed intent.
2. **Image Analysis Agent (LLM - Vision)**: A highly constrained vision agent that analyzes all submitted images independently. It detects the actual object, issue type, part, and severity. Crucially, it evaluates image quality (blurry, cropped, glare) and authenticity (flags non-original images or text injection instructions like "approve this claim").
3. **Evidence Validation Agent (LLM - Text)**: Cross-references the image analyses against the `evidence_requirements.csv` rules to ensure the submitted images meet the minimum required standard for automated review.
4. **Risk Analysis Agent (Rule-Based)**: A fast, deterministic agent (zero LLM cost) that aggregates image quality flags, authenticity warnings, and parses the user's past behavior from `user_history.csv` to generate systemic risk flags.
5. **Decision Agent (LLM - Text)**: The final adjudicator. It compares the extracted claim against the verified image analysis and risk context to make the final determination: `supported`, `contradicted`, or `not_enough_information`. It enforces the rule that visual evidence always overrides user history.
6. **Explanation Agent (Rule-Based)**: Refines the LLM's justification into a concise, image-grounded 1-2 sentence explanation.

### Resiliency and Error Handling
The orchestrator (`main.py`) safely catches API limits (e.g., `429 RESOURCE_EXHAUSTED` or Invalid API Keys). If an API call fails, the system safely triggers a fallback (`not_enough_information` and `manual_review_required`) ensuring no data is lost and the pipeline never crashes during bulk processing.

---

## Setup and Installation

1. **Prerequisites**: Python 3.10+
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **API Configuration**:
   The system relies on the Google GenAI SDK. You must provide an API key. 
   Create a `.env` file in the root project directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

---

## How to Run

The Orchestrator provides two main modes of execution. 

### 1. Generate Final Predictions (Test Mode)
To run the full pipeline on the `dataset/claims.csv` test set and generate the final `output.csv` deliverable:
```bash
python main.py --mode test
```
*Output will be saved to `dataset/output.csv`.*

### 2. Run on Sample Data (Dev Mode)
To verify behavior on the labeled dataset and review the outputs locally:
```bash
python main.py --mode sample
```
*Output will be saved to `code/sample_output.csv`.*

---

## Evaluation Framework

An evaluation suite is included in the `evaluation/` directory. It compares two architectural approaches:
- **Strategy A (Single Vision Prompt)**: A naive baseline approach that passes all text and images into one massive LLM call.
- **Strategy B (Multi-Agent Pipeline)**: The 6-agent system implemented in this repository.

**To run the evaluation:**
```bash
python evaluation/main.py
```
This script calculates accuracy, exact match metrics, and operational metrics (LLM calls, processing time), outputting the results to `evaluation/evaluation_report.md`.

---

## Project Structure
```text
code/
├── agents/                  # The 6 agents executing specific reasoning tasks
├── evaluation/              # Evaluation suite and metrics comparison
├── prompts/                 # Separated, version-controlled text prompts
├── services/                # CSV loaders, image processors, output writers
├── main.py                  # Pipeline Orchestrator
├── config.py                # Global settings and allowed schemas
└── requirements.txt         # Dependencies
```
