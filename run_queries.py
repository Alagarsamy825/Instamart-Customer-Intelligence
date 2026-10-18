"""Run the seven queries in sql/queries.sql against data/instamart.db.

    python run_queries.py
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

#-- name: Q1
EXPECTED = [f"Q{i}" for i in range(1, 8)]
# Matches "-- name: Q1" as well as "-- Q1" or "-- Q1 - WHERE then HAVING".
HEADER = re.compile(r"^--\s*(?:name:\s*)?(Q\d+)\b[^\n]*$", re.MULTILINE | re.IGNORECASE)


def load_queries(path):
    """Split the SQL file on query header comments into {name: sql}."""
    text = path.read_text().replace("\r\n", "\n")
    parts = HEADER.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    queries = {
        name.upper(): body.strip() for name, body in zip(parts[1::2], parts[2::2])
    }

    missing = [name for name in EXPECTED if name not in queries]
    if missing:
        raise SystemExit(
            f"{path} has no header comment for {', '.join(missing)}. "
            "Put a line like '-- Q1' above each query."
        )
    return queries


def run_all():
    queries = load_queries(SQL_PATH)
    results = {}
    with sqlite3.connect(DB_PATH) as conn:
        for name, sql in queries.items():
            results[name] = pd.read_sql_query(sql, conn)
    return results


def report(r):
    print("Q1 customers:", len(r["Q1"]), "(reference 30)")

    q2 = r["Q2"]
    top = q2.iloc[0]
    print(f"Q2 rows: {len(q2)} (reference 30)")
    print(f"Q2 total: INR {q2['revenue'].sum():,.2f} (reference 2,998,668.80)")
    print(f"Q2 largest cell: {top['city']} / {top['category']} INR {top['revenue']:,.2f}")

    print("Q3 products:", len(r["Q3"]), "(reference 11)")
    print("Q4 customers:", len(r["Q4"]), "(reference 25)")

    q5 = r["Q5"]
    print(f"Q5 customers: {len(q5)} (reference 550), average INR {q5['avg_spend'].iloc[0]:,.2f}")

    q6 = r["Q6"]
    print("\nQ6 top customer per city:")
    print(q6[q6["spend_row_number"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
    differing = int((q6["spend_rank"] != q6["spend_row_number"]).sum())
    print(f"Q6 rows where RANK differs from ROW_NUMBER: {differing} (reference 8)")

    q7 = r["Q7"]
    print(f"\nQ7 daily rows: {len(q7)} (reference 183)")
    print(f"Q7 last cumulative: INR {q7['cumulative_revenue'].iloc[-1]:,.2f}")


if __name__ == "__main__":
    report(run_all())