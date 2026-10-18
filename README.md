# Instamart Customer Intelligence — Part A: Data Foundation

Audit the raw exports, clean them with documented rules, store them in SQLite, and answer business questions in SQL and pandas. All amounts are in INR.

## Reproduce

git clone <repo-url> && cd Instamart-Customer-Intelligence
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# copy the two supplied files (including generate_data.py) into the project root, then:
python generate_data.py     # writes data/raw/*.csv
python audit_data.py        # audit + clean -> data/clean/*.csv and data/instamart.db
python validate.py          # must print ALL CHECKPOINTS PASSED
python run_queries.py       # runs sql/queries.sql against the database
python explore.py           # pandas cross-check and exploratory statistics
```

## Project layout

| Path | Purpose |
|---|---|
| `generate_data.py` | Supplied script that creates the raw exports in `data/raw/` |
| `audit_data.py` | Audit, cleaning rules, writes CSVs and `data/instamart.db` |
| `validate.py` | Asserts the reference checkpoints |
| `sql/queries.sql` | The seven analysis queries (Q1–Q7) |
| `run_queries.py` | Runs each query with `pd.read_sql_query` and prints the results |
| `explore.py` | pivot_table cross-check, summary statistics, correlation |

## Raw data audit

| Table | Metric | Value |
|---|---|---|
| orders | Raw rows | 13,084 |
| orders | Exact duplicate rows | 119 |
| orders | Pending rows | 656 |
| orders | Distinct `order_id` | 12,309 |
| orders | Non-numeric `order_amount` | 120 |
| orders | Negative `delivery_minutes` | 106 |
| orders | Blank `delivery_minutes` | 2,541 |
| customers | Blank `age` | 61 |
| customers | `age` outside 10–100 | 6 |
| customers | Distinct raw city spellings | 24 |
| customers | Distinct non-blank raw gender spellings | 10 |
| customers | Blank `gender` | 30 |

## Cleaning rules

### Customers

| # | Rule | Rows / values changed |
|---|---|---|
| 1 | Strip leading and trailing spaces from `city` and convert to title case | 24 raw spellings reduced to 6 cities |
| 2 | Map every `gender` spelling to `male`, `female` or `unknown`; blanks become `unknown` | 10 raw spellings reduced to 3 values; 30 blanks set to `unknown` |
| 3 | Treat `age` outside 10–100 as missing | 6 values set to missing |
| 4 | Fill every missing `age` with the median age of the customer's city (`groupby` + `transform`) | 67 values filled (61 blank + 6 out of range) |
| 5 | Convert `signup_date` to datetime | All rows |

### Orders (applied in this order)

| # | Rule | Rows / values changed |
|---|---|---|
| 1 | Convert `order_ts` to datetime | All rows |
| 2 | Remove exact duplicate rows | 119 rows removed |
| 3 | Keep only the record with the latest `order_ts` for each `order_id` | `<N>` older records removed, leaving 12,309 rows (one per order) |
| 4 | Convert `order_amount` to numeric (unparseable becomes missing), then fill missing amounts with `round(price_inr × quantity × (1 − discount_pct / 100), 2)` | `<N>` amounts filled |
| 5 | Set negative `delivery_minutes` to missing | `<N>` values set to missing |
| 6 | For `completed` orders, fill missing `delivery_minutes` with the median of the product category among completed orders (`groupby` + `transform`) | `<N>` values filled |
| 7 | Leave `delivery_minutes` missing for `cancelled` orders: they were never delivered, so a filled value would be invented data | 1,199 values left missing |

> Fill each `<N>` from the "ORDER CLEANING - what each rule changed" block that `clean_data.py` prints. Those counts are taken after the earlier rules have run, so they can differ slightly from the raw audit figures above.

### Result after cleaning

| Check | Value |
|---|---|
| Order rows | 12,309 (one per `order_id`) |
| Completed / cancelled / pending | 11,110 / 1,199 / 0 |
| Missing `order_amount` | 0 |
| Missing `delivery_minutes` | 0 among completed, 1,199 among cancelled |
| Completed revenue | INR 2,998,668.80 |
| Customer cities | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 |
| Customer genders | female 610, male 560, unknown 30 |
| Customer ages | 0 missing, all between 18 and 60 |

## Storage

The clean tables are written to `data/clean/` as CSV files and to `data/instamart.db` as the tables `customers`, `products` and `orders`. `order_ts` is stored as text in the form `YYYY-MM-DD HH:MM:SS`.

`validate.py` asserts the reference checkpoints above and prints `ALL CHECKPOINTS PASSED` when they all hold.

## SQL analysis

All queries are in `sql/queries.sql`. Only completed orders count as revenue or spend.

| Query | Technique | Result |
|---|---|---|
| Q1 | `WHERE` then `HAVING` | 30 customers |
| Q2 | Three-table join | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 | Scalar subquery | 11 products priced above the average |
| Q4 | `LEFT JOIN` | 25 customers with no completed order |
| Q5 | Common table expressions | 550 customers above the average spend of INR 2,552.06 |
| Q6 | `RANK`, `DENSE_RANK`, `ROW_NUMBER` | Top customer per city listed below; 8 rows where `RANK()` differs from `ROW_NUMBER()` |
| Q7 | Running total window | 183 daily rows; final cumulative revenue INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

**Why `RANK()` and `ROW_NUMBER()` differ:** when customers in a city tie on total spend, `RANK()` gives them the same rank while `ROW_NUMBER()` still gives each a unique, increasing number, so every row after a tie gets a different value from the two functions.

## pandas cross-check and exploratory analysis

- A `pd.pivot_table(..., margins=True)` rebuild of Q2 gives a grand total of INR 2,998,668.80, equal to the SQL total.
- Per-city completed order amounts (`groupby` with named aggregation): Bengaluru sum INR 523,643.40, mean INR 276.47, count 1,894; Mumbai sum INR 479,058.50, mean INR 297.00, count 1,613.

### Distribution of completed `order_amount`

| Statistic | Value |
|---|---|
| Mean / median | INR 269.91 / INR 135.00 |
| Skewness | about 4.61 |
| Q1 / Q3 | INR 64.80 / INR 297.00 |
| Upper fence (Q3 + 1.5 × IQR) | INR 645.30 |
| Outliers above / below fences | 897 / 0 |

### Four questions (max 200 words total)

**What is this data about?** Instamart quick-commerce orders: 11,110 completed orders across six cities, with product category, quantity, discount and delivery time.

**What do the distributions look like?** Order amounts are strongly right-skewed. The mean (INR 269.91) is double the median (INR 135.00), skewness is about 4.61, and the middle half of orders lies between INR 64.80 and INR 297.00. Most baskets are small, with a long tail of expensive ones.

**Are there outliers or anomalies?** The upper fence is INR 645.30 and 897 orders exceed it; none fall below the lower fence. They look like genuine high-value purchases rather than errors, so they are kept, but averages are pulled upward.

**What relationships exist?** All correlations are weak. The largest in absolute value is `order_amount` with `discount_pct` at about −0.095, just ahead of `order_amount` with `quantity` at +0.087. A correlation of this size does not establish causation, and it is too small to say that discounts drive order value at all.

