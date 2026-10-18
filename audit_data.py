import sqlite3
from pathlib import Path
 
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"
 
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
GENDER_MAP = {"male": "male", "m": "male", "female": "female", "f": "female"}

# Loading the raw CSV files into pandas DataFrames
def load_raw():
    """Read the four raw CSVs. Tickets keep empty texts as empty strings."""
    return {
        "orders": pd.read_csv(RAW_DIR / "orders.csv"),
        "customers": pd.read_csv(RAW_DIR / "customers.csv"),
        "tickets": pd.read_csv(RAW_DIR / "tickets.csv", keep_default_na=False),
        "products": pd.read_csv(RAW_DIR / "products.csv"),
    }

# Audit and clean the data
def profile_table(name, df):
    """Print shape, dtypes and missing values (count and %) for every column."""
    print(f"\n{'=' * 50}\nTABLE: {name}\n{'=' * 50}")
    print("Shape:", df.shape)
 
    print("\nData types:")
    print(df.dtypes)
 
    missing = df.isna().sum()
    report = pd.DataFrame(
        {
            "Missing Count": missing,
            "Missing Percentage": (missing / len(df) * 100).round(2),
        }
    )
    print("\nMissing values by column:")
    print(report)

def audit_orders(orders):
    amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
 
    return {
        "Raw order rows": len(orders),
        "Exact duplicate rows": int(orders.duplicated().sum()),
        "Pending rows": int((orders["status"] == "pending").sum()),
        "Distinct order IDs": int(orders["order_id"].nunique()),
        "Non-numeric order amounts": int(amounts.isna().sum()),
        "Negative delivery minutes": int((delivery < 0).sum()),
        "Blank delivery minutes": int(orders["delivery_minutes"].isna().sum()),
    }

def audit_customers(customers):
    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"]
 
    return {
        "Blank ages": int(customers["age"].isna().sum()),
        "Ages outside 10-100": int(((age < 10) | (age > 100)).sum()),
        "Distinct raw city spellings": int(customers["city"].nunique()),
        "Distinct non-blank raw gender spellings": int(gender.nunique()),
        "Blank genders": int(gender.isna().sum()),
    }

def print_audit(title, audit):
    print(f"\n{title}")
    print(pd.DataFrame(audit.items(), columns=["Metric", "Count"]).to_string(index=False))

# Cleaning functions for customers and orders

def clean_customers(raw):
    df = raw.copy()
 
    # (a) Trim spaces and title-case the city so only six cities remain.
    df["city"] = df["city"].str.strip().str.title()
 
    # (b) One spelling per gender; blanks and anything unexpected become unknown.
    df["gender"] = (
        df["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
    )
 
    # (c) Ages outside 10-100 are treated as missing, then every missing age
    #     gets the median age of that customer's city.
    age = pd.to_numeric(df["age"], errors="coerce")
    age = age.mask((age < 10) | (age > 100))
    df["age"] = age.groupby(df["city"]).transform(lambda s: s.fillna(s.median()))
 
    # (d) Proper datetime for signup_date.
    df["signup_date"] = pd.to_datetime(df["signup_date"])
 
    return df

 
def clean_orders(raw, products):
    """Apply the order cleaning rules in the required order.
 
    Returns the clean table and a dict with how many rows/values each rule changed
    (handy for the README).
    """
    df = raw.copy()
    changes = {}
 
    # 1. order_ts -> datetime
    df["order_ts"] = pd.to_datetime(df["order_ts"])
 
    # 2. Remove exact duplicate rows
    rows_before = len(df)
    df = df.drop_duplicates()
    changes["exact duplicate rows removed"] = rows_before - len(df)
 
    # 3. Keep only the latest record (by order_ts) for each order_id
    rows_before = len(df)
    latest_rows = df.groupby("order_id")["order_ts"].idxmax()
    df = df.loc[latest_rows].reset_index(drop=True)
    changes["older duplicate order_id records removed"] = rows_before - len(df)
 
    # Bring in price and category; both are needed for the next steps.
    df = df.merge(
        products[["product_id", "price_inr", "category"]],
        on="product_id",
        how="left",
    )
 
    # 4. order_amount -> numeric, then fill gaps from price x quantity x discount
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    changes["order_amount filled from product price"] = int(df["order_amount"].isna().sum())
 
    price_based_amount = (
        df["price_inr"] * df["quantity"] * (1 - df["discount_pct"] / 100)
    ).round(2)
    df["order_amount"] = df["order_amount"].fillna(price_based_amount)
 
    # 5. delivery_minutes -> numeric, negatives become missing
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
 
    negative = df["delivery_minutes"] < 0
    changes["negative delivery_minutes set to missing"] = int(negative.sum())
    df.loc[negative, "delivery_minutes"] = np.nan
 
    # Completed orders: fill with the median of their product category
    # (computed among completed orders only). Cancelled orders stay missing,
    # because they were never delivered.
    completed = df["status"] == "completed"
    category_median = (
        df.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    )
    missing_before = int(df.loc[completed, "delivery_minutes"].isna().sum())
    df.loc[completed, "delivery_minutes"] = df.loc[completed, "delivery_minutes"].fillna(
        category_median
    )
    changes["completed delivery_minutes filled with category median"] = missing_before
 
    # price_inr and category were only helpers, so the clean table keeps the
    # original order columns.
    df = df.drop(columns=["price_inr", "category"])
 
    return df, changes

 # Checks and storage
# --------------------------------------------------------------------------
def print_orders_check(orders):
    completed = orders["status"] == "completed"
    cancelled = orders["status"] == "cancelled"
 
    print("\n--- Orders cleaning check ---")
    print("Total rows:", len(orders))
    print("Unique order IDs:", orders["order_id"].nunique())
    print("\nOrders by status:")
    print(orders["status"].value_counts())
    print("\nPending orders:", int((orders["status"] == "pending").sum()))
    print("Missing order_amount:", int(orders["order_amount"].isna().sum()))
    print(
        "Missing delivery_minutes (completed):",
        int(orders.loc[completed, "delivery_minutes"].isna().sum()),
    )
    print(
        "Missing delivery_minutes (cancelled):",
        int(orders.loc[cancelled, "delivery_minutes"].isna().sum()),
    )
    total = orders.loc[completed, "order_amount"].sum()
    print(f"Completed orders total: INR {total:,.2f}")
 
 
def print_customers_check(customers):
    print("\n--- Customers cleaning check ---")
    print(customers["city"].value_counts())
    print(customers["gender"].value_counts())
    print("Missing ages:", int(customers["age"].isna().sum()))
    print("Age range:", customers["age"].min(), "to", customers["age"].max())
 
 
def save_clean_tables(orders, customers, products):
    """Write CSVs to data/clean/ and tables to the SQLite database."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
 
    # Store order_ts as plain text so CSV and SQLite agree.
    orders_out = orders.copy()
    orders_out["order_ts"] = orders_out["order_ts"].dt.strftime(TS_FORMAT)
 
    tables = {"orders": orders_out, "customers": customers, "products": products}
 
    for name, df in tables.items():
        df.to_csv(CLEAN_DIR / f"{name}.csv", index=False)
 
    with sqlite3.connect(DB_PATH) as conn:
        for name, df in tables.items():
            df.to_sql(name, conn, if_exists="replace", index=False)
 
    print(f"\nSaved clean CSVs to {CLEAN_DIR}")
    print(f"Saved SQLite database to {DB_PATH}")
 
 
# Main
def main():
    raw = load_raw()
 
    for name in ("orders", "customers", "tickets"):
        profile_table(name, raw[name])
 
    print_audit("ORDER AUDIT", audit_orders(raw["orders"]))
    print_audit("CUSTOMER AUDIT", audit_customers(raw["customers"]))
 
    clean_order_table, order_changes = clean_orders(raw["orders"], raw["products"])
    clean_customer_table = clean_customers(raw["customers"])
 
    print("\nORDER CLEANING - what each rule changed")
    for rule, count in order_changes.items():
        print(f"  {rule}: {count}")
 
    print_orders_check(clean_order_table)
    print_customers_check(clean_customer_table)
 
    save_clean_tables(clean_order_table, clean_customer_table, raw["products"])
 
 
if __name__ == "__main__":
    main()
