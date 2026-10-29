"""Check the clean data against the reference checkpoints from the brief."""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

orders = pd.read_csv(CLEAN_DIR / "orders.csv")
customers = pd.read_csv(CLEAN_DIR / "customers.csv")

completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"
city_counts = customers["city"].value_counts()
gender_counts = customers["gender"].value_counts()

# Orders
assert len(orders) == 12_309, "orders row count"
assert orders["order_id"].nunique() == 12_309, "one row per order_id"
assert completed.sum() == 11_110, "completed orders"
assert cancelled.sum() == 1_199, "cancelled orders"
assert (orders["status"] == "pending").sum() == 0, "no pending rows"
assert orders["order_amount"].isna().sum() == 0, "no missing order_amount"
assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0, "completed delivery filled"
assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1_199, "cancelled delivery left missing"
assert round(orders.loc[completed, "order_amount"].sum(), 2) == 2_998_668.80, "completed revenue"

# Customers
assert customers["city"].nunique() == 6, "six cities"
assert city_counts["Pune"] == 227 and city_counts["Delhi"] == 210, "city counts"
assert gender_counts["female"] == 610, "female count"
assert gender_counts["male"] == 560, "male count"
assert gender_counts["unknown"] == 30, "unknown gender count"
assert customers["age"].isna().sum() == 0, "no missing ages"
assert customers["age"].between(18, 60).all(), "ages between 18 and 60"

# Database
with sqlite3.connect(DB_PATH) as conn:
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {"customers", "products", "orders"} <= tables, "database tables"
    db_total = conn.execute(
        "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'"
    ).fetchone()[0]
    assert db_total == 2_998_668.80, "database completed revenue"

print("ALL CHECKPOINTS PASSED")