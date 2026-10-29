# Instamart Customer Intelligence

> An educational implementation of a fictional Instamart-style customer intelligence and support workflow. Use the supplied project data and assignment brief as the source of truth for validation. Do not add real customer data or secrets to the repository.

## Overview

This project analyzes customer, product, order, and support-ticket data across four parts:

- **Part A — Data Foundation:** audit and clean raw CSV files, persist cleaned data, and run SQL and pandas analysis.
- **Part B — Customer Segments and Lapse Prediction:** calculate RFM metrics, group customers with K-means, train a lapse prediction model, and create a retention list.
- **Part C — Support Ticket Triage:** validate tickets, redact personally identifiable information (PII), classify tickets, handle transient errors, and route tickets to the appropriate team or human review.
- **Part D — Policy RAG and Refund Agent:** retrieve relevant policy passages, generate grounded answers, evaluate retrieval, and exercise bounded refund-agent tools where the implementation is available.

## Reproduce

Python 3.11 or later is recommended.

git clone <your-public-repository-url>
cd Instamart-Customer-Intelligence

python3 -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Generate synthetic input data only if it is not already supplied
python generate_data.py

# Run the project entry point
python run_project.py


Run commands from the repository root. If a stage fails, read the error printed in the terminal before continuing. Review the generated artifacts in `outputs/` and the cleaned data in `data/clean/`. Do not commit the virtual environment.

## Part A — Data Foundation

Part A prepares raw customer, product, and order data for analysis.

- Audits the source CSV files for data quality issues.
- Cleans inconsistent values, duplicate orders, missing values, and invalid data according to the implemented rules.
- Writes cleaned CSV files to `data/clean/`.
- Stores cleaned tables in the SQLite database `data/instamart.db`.
- Runs SQL queries and pandas analysis to answer business questions.

**Main outputs:** cleaned CSV files under `data/clean/` and `data/instamart.db`.

The SQL queries are maintained in `sql/queries.sql`. Use the assignment brief and the actual run outputs to validate the reported counts and statistics; do not hardcode expected values to force a match.

## Part B — Customer Segments and Lapse Prediction

Part B uses the cleaned data from Part A to analyze customer behaviour.

- Calculates RFM metrics: Recency, Frequency, and Monetary value.
- Groups customers using K-means clustering.
- Trains a model to predict which customers may stop ordering.
- Produces a retention list to support customer outreach decisions.

The lapse model should use the time-based feature and label definitions specified in the project implementation and assignment brief. Treat the retention list as a decision-support artifact, not a guarantee that a customer will lapse.

**Main outputs:** `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`.

## Part C — Support Ticket Triage

Part C classifies support tickets and routes them to the appropriate destination.

- Validates tickets before processing.
- Redacts email addresses, phone numbers, and order IDs before model prompting.
- Compares zero-shot and few-shot prompting.
- Retries temporary model errors and validates model responses.
- Applies token-budget and output guardrails.
- Routes tickets to the appropriate support team or human review.

**Main output:** the ticket-triage CSV generated in `outputs/`. Check the Part C source code for the exact filename used by this repository.

## Part D — Policy RAG and Refund Agent

Part D retrieves policy information and demonstrates a bounded refund-agent workflow.

- Splits policy documents into chunks, including the configured overlap.
- Uses TF-IDF representations and cosine similarity to rank relevant policy passages.
- Generates answers grounded in retrieved policy context using the project's configured model implementation.
- Evaluates retrieval with the overlap settings used by the Part D scripts.
- Provides refund-agent tools for order lookup, policy search, refund calculation, refund issuance, and escalation where implemented.
- Uses local persistence and audit artifacts for the simulated agent workflow.

The supplied teaching implementation describes these refund rules: completed orders delivered in over **45 minutes** qualify for a full refund, while orders over **INR 1000** are escalated instead of being automatically refunded. Confirm these rules against the actual Part D source before treating them as the final acceptance criteria.

**Expected evaluation outputs:** `outputs/policy_eval_overlap0.csv` and `outputs/policy_eval_overlap10.csv`.

## Limitations and Next Steps

- This is an educational project, not a production customer-support or payment system.
- Use synthetic or appropriately anonymized data only. Do not commit credentials, API keys, or personal customer information.
- The configured mock model may be a deterministic, rule-based simulation rather than a hosted language model. Confirm the implementation in `mock_llm.py`.
- The refund ledger and refund tools are a local simulation; they do not move money or integrate with a payment provider.
- Verify that every required Part A–D source file and every artifact listed in the assignment's **Required Repository Contents** is committed.
- Compare actual outputs with the assignment's numeric checkpoints and report the measured results. Do not invent or hardcode metrics that have not been produced by a successful run.
- For production use, add appropriate access controls, secure secret management, monitoring, schema validation, data lineage, model-drift checks, and human approval workflows.
