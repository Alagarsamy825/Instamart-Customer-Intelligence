"""Part B, tasks 1-3: RFM table, K-means customer segments, PCA projection.

Run after clean_data.py:
    python rfm_segments.py

Output:
    outputs/customer_segments.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ROOT = Path(__file__).resolve().parent
CLEAN_DIR = Path("data") / "clean"
OUTPUT_DIR = Path("outputs")

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
RANDOM_STATE = 42
CHOSEN_K = 4

# Business name and one action per segment type. Cluster numbers are only
# labels, so names are given by looking at each segment's averages.
SEGMENT_INFO = {
    "Premium Big Spenders": "Personal offers and early access; protect them with a loyalty tier.",
    "Loyal Regulars": "Reward with points and cross-sell other categories.",
    "Slipping Away": "Send a win-back coupon before they are gone for good.",
    "Occasional Shoppers": "Nudge towards a second order with a free-delivery threshold.",
}

# Task 1: RFM table
def build_rfm(orders):
    """One row per customer with at least one completed order."""
    completed = orders[orders["status"] == "completed"]

    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)

    return rfm[["recency", "frequency", "monetary"]].reset_index()

# Task 2: K-means for K = 2..8
def k_means_table(scaled):
    rows = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        rows.append(
            {
                "K": k,
                "inertia": round(model.inertia_, 1),
                "silhouette": round(silhouette_score(scaled, labels), 3),
            }
        )
    return pd.DataFrame(rows)

# Task 3: profile and name the segments
def profile_segments(rfm):
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        recency=("recency", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    )
    return profile.round(2)


def name_segments(profile):
    """Return {segment_id: name} using the segment averages, not the id numbers."""
    names = {}
    remaining = profile.copy()

    # The segment with the highest mean monetary value is the Premium one.
    premium_id = remaining["monetary"].idxmax()
    names[premium_id] = "Premium Big Spenders"
    remaining = remaining.drop(premium_id)

    if CHOSEN_K != 4:
        # Naming rules below are written for four segments.
        for rank, seg_id in enumerate(remaining["monetary"].sort_values(ascending=False).index, 1):
            names[seg_id] = f"Segment {rank}"
        return names

    # Longest time since the last order -> at risk of being lost.
    slipping_id = remaining["recency"].idxmax()
    names[slipping_id] = "Slipping Away"
    remaining = remaining.drop(slipping_id)

    # Of the two left, the one that orders most often is the loyal group.
    loyal_id = remaining["frequency"].idxmax()
    names[loyal_id] = "Loyal Regulars"
    remaining = remaining.drop(loyal_id)

    names[remaining.index[0]] = "Occasional Shoppers"
    return names


def main():
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])

    # ---- Task 1
    rfm = build_rfm(orders)
    print(f"Customers with a completed order: {len(rfm)}")
    print(f"Recency   median {rfm['recency'].median():.0f}, min {rfm['recency'].min()}, max {rfm['recency'].max()}")
    print(f"Frequency mean {rfm['frequency'].mean():.2f}, max {rfm['frequency'].max()}")
    print(f"Monetary  mean {rfm['monetary'].mean():,.2f}, max {rfm['monetary'].max():,.2f}")

    scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])

    # ---- Task 2
    print("\nK-means: inertia and silhouette for K = 2..8")
    print(k_means_table(scaled).to_string(index=False))

    # The final K comes from the table above: inertia drops sharply up to K=4
    # and flattens afterwards (the elbow), K=4 has the best silhouette, and four
    # segments are a number the business can act on.
    final_model = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=RANDOM_STATE)
    rfm["segment_id"] = final_model.fit_predict(scaled)

    # ---- Task 3 (a) and (b)
    profile = profile_segments(rfm)
    names = name_segments(profile)
    premium_id = profile["monetary"].idxmax()

    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium_id

    profile["segment_name"] = profile.index.map(names)
    profile["action"] = profile["segment_name"].map(SEGMENT_INFO)
    print(f"\nSegment profile for K = {CHOSEN_K} (cluster numbers are labels only, they have no order):")
    print(profile.to_string())

    # ---- Task 3 (c)
    pca = PCA(n_components=2).fit(scaled)
    ratios = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: {ratios[0]:.3f} + {ratios[1]:.3f} = {ratios.sum():.3f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    columns = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    rfm[columns].to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'customer_segments.csv'}")


if __name__ == "__main__":
    main()
