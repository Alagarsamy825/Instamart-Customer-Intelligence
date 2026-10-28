Instamart Customer Intelligence

This project analyzes Instamart customer and order data in three parts.

What each part does

Part A — Data Foundation
Part A prepares the raw data so it can be used reliably.
* Audits the original customer, product, and order CSV files.
* Cleans inconsistent values, duplicate orders, missing values, and invalid data.
* Saves cleaned CSV files in data/clean/.
* Stores the cleaned tables in data/instamart.db (SQLite).
* Runs SQL queries and pandas analysis to answer business questions.
Main outputs: cleaned CSV files and data/instamart.db.

Part B — Customer Segments and Lapse Prediction
Part B uses the cleaned data from Part A to understand customer behaviour.
* Calculates RFM values: Recency, Frequency, and Monetary value.
* Groups customers into segments using K-means clustering.
* Builds a model to predict which customers may stop ordering.
* Creates a retention list to help the business decide which customers to contact.
Main outputs: outputs/customer_segments.csv, outputs/lapse_model.joblib, and outputs/retention_list.csv.

Part C — Support Ticket Triage
Part C classifies customer support tickets and decides where each ticket should go.
* Checks tickets before sending them to the model.
* Redacts email addresses, phone numbers, and order IDs.
* Compares zero-shot and few-shot prompts.
* Retries temporary model errors and validates model responses.
* Applies a token budget and sends tickets to the appropriate team or human review.
Main output: the final ticket-triage CSV in outputs/ (the exact filename is set by the Part C script).