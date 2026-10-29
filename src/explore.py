"""pandas cross-check of the SQL results plus exploratory statistics."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "clean"

orders = pd.read_csv(CLEAN_DIR / "orders.csv")
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")

completed = orders[orders["status"] == "completed"]

# (a) Rebuild Q2 with a pivot table and compare with the SQL total

merged = (
    completed
    .merge(customers[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id")
)

pivot = pd.pivot_table(
    merged,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Total",
).round(2)

print("Revenue by city and category (INR):")
print(pivot)

grand_total = pivot.loc["Total", "Total"]
print(f"\nPivot grand total: INR {grand_total:,.2f} (SQL total: INR 2,998,668.80)")
assert round(grand_total, 2) == 2_998_668.80

city_summary = (
    merged.groupby("city")
    .agg(
        total=("order_amount", "sum"),
        average=("order_amount", "mean"),
        orders=("order_amount", "count"),
    )
    .round(2)
)
print("\nCompleted orders per city:")
print(city_summary)

# --------------------------------------------------------------------------
# (b) Distribution of completed order_amount
# --------------------------------------------------------------------------
amount = completed["order_amount"]
q1, q3 = amount.quantile(0.25), amount.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\nOrder amount statistics (INR):")
print(f"  mean      {amount.mean():.2f}")
print(f"  median    {amount.median():.2f}")
print(f"  skewness  {amount.skew():.2f}")
print(f"  Q1 / Q3   {q1:.2f} / {q3:.2f}")
print(f"  fences    {lower_fence:.2f} / {upper_fence:.2f}")
print(f"  outliers below lower fence: {int((amount < lower_fence).sum())}")
print(f"  outliers above upper fence: {int((amount > upper_fence).sum())}")

# Correlation among the numeric columns, then the strongest pair
numeric_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = completed[numeric_cols].corr()
print("\nCorrelation matrix:")
print(corr.round(3))

pairs = corr.where(~pd.DataFrame(
    [[i >= j for j in range(len(numeric_cols))] for i in range(len(numeric_cols))],
    index=corr.index, columns=corr.columns,
)).stack()
strongest = pairs.abs().idxmax()
print(f"\nStrongest pair: {strongest[0]} and {strongest[1]} ({pairs[strongest]:+.3f})")