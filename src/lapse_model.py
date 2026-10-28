from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ROOT = Path(__file__).resolve().parent
CLEAN_DIR = Path("data") / "clean"
OUTPUT_DIR = Path("outputs")

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")  # the lapse window is all of September
RANDOM_STATE = 42

NUMERIC = [
    "recency", "frequency", "monetary", "avg_discount_pct",
    "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days",
]
CATEGORICAL = ["city", "gender", "favourite_category"]

# Task 4: cohort, label and features
def build_cohort(orders, customers, products):
    """One row per customer with a completed order before the cutoff.

    Features use only orders dated before the cutoff. The label looks at September.
    """
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    before = orders[orders["order_ts"] < CUTOFF]
    completed = before[before["status"] == "completed"]
    cancelled = before[before["status"] == "cancelled"]

    cohort = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
        favourite_category=("category", lambda s: s.mode().iloc[0]),
    )
    cohort["recency"] = (CUTOFF - cohort["last_order_ts"]).dt.days

    # Cancelled orders divided by all orders that reached a final status.
    n_cancelled = cancelled.groupby("customer_id").size().reindex(cohort.index, fill_value=0)
    cohort["cancellation_rate"] = n_cancelled / (cohort["frequency"] + n_cancelled)

    cohort = cohort.reset_index().merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]],
        on="customer_id",
        how="left",
    )
    cohort["tenure_days"] = (CUTOFF - cohort["signup_date"]).dt.days

    # Label: lapsed = no completed order anywhere in September.
    september = orders[
        (orders["status"] == "completed")
        & (orders["order_ts"] >= CUTOFF)
        & (orders["order_ts"] < WINDOW_END)
    ]
    cohort["lapsed"] = (~cohort["customer_id"].isin(september["customer_id"])).astype(int)

    return cohort

# Task 5: preprocessing (fitted on training rows only, inside the Pipeline)
def build_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def make_pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])

# Helpers
def lapsed_scores(y_true, y_pred):
    """Precision, recall and F1 of the lapsed (positive) class."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def show_test_result(name, model, X_test, y_test):
    predicted = model.predict(X_test)
    scores = lapsed_scores(y_test, predicted)
    print(f"\n{name}")
    print("Confusion matrix (rows = actual 0/1, columns = predicted 0/1):")
    print(confusion_matrix(y_test, predicted))
    print(f"precision {scores['precision']:.3f}  recall {scores['recall']:.3f}  F1 {scores['f1']:.3f}")
    return scores


def oversample_lapsed(X, y):
    """Duplicate random lapsed rows (with replacement) until both classes match."""
    gap = int((y == 0).sum() - (y == 1).sum())
    extra = X[y == 1].sample(n=gap, replace=True, random_state=RANDOM_STATE)
    X_balanced = pd.concat([X, extra])
    y_balanced = pd.concat([y, pd.Series(1, index=extra.index)])
    return X_balanced, y_balanced


def main():
    orders = pd.read_csv(CLEAN_DIR / "orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(CLEAN_DIR / "customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv(CLEAN_DIR / "products.csv")

    # ---- Task 4: cohort and split
    cohort = build_cohort(orders, customers, products)
    X = cohort[NUMERIC + CATEGORICAL]
    y = cohort["lapsed"]

    print(f"Cohort: {len(cohort)} customers, {y.sum()} lapsed ({y.mean():.2%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)}, test rows: {len(X_test)}")

    # ---- Task 5: no missing values, and what the scaler learned
    assert X.isna().sum().sum() == 0, "a feature column has missing values"
    print("No missing values in any feature column.")

    scaler = build_preprocessor().fit(X_train).named_transformers_["num"]
    train_mean = scaler.mean_[NUMERIC.index("frequency")]
    print(f"Mean frequency learned by the scaler (train only): {train_mean:.3f}")
    print(f"Mean frequency of the full cohort:                 {X['frequency'].mean():.3f}")
    print("They differ because the scaler only saw the 80% training rows, never the test rows.")

    # ---- Task 6 (a): baseline
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    print(f"\nBaseline (always 'not lapsed'): accuracy {accuracy_score(y_test, baseline_pred):.3f}, "
          f"lapsed F1 {f1_score(y_test, baseline_pred, zero_division=0):.3f}")

    # ---- Task 6 (b) and (c): tuning with 5-fold CV on the training rows only
    logistic_search = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000)),
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5, scoring="f1", n_jobs=-1,
    ).fit(X_train, y_train)

    forest_search = GridSearchCV(
        make_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        param_grid={"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
        cv=5, scoring="f1", n_jobs=-1,
    ).fit(X_train, y_train)

    print(f"\nLogistic regression: best {logistic_search.best_params_}, CV F1 {logistic_search.best_score_:.3f}")
    print(f"Random forest:       best {forest_search.best_params_}, CV F1 {forest_search.best_score_:.3f}")

    # ---- Task 6 (d): the test set is used only now, after tuning
    logistic_scores = show_test_result("Logistic regression on the test set", logistic_search.best_estimator_, X_test, y_test)
    forest_scores = show_test_result("Random forest on the test set", forest_search.best_estimator_, X_test, y_test)

    # The model is picked by cross-validated F1, so the test set stays a clean check.
    if forest_search.best_score_ >= logistic_search.best_score_:
        selected_name, selected = "Random forest", forest_search.best_estimator_
    else:
        selected_name, selected = "Logistic regression", logistic_search.best_estimator_
    print(f"\nSelected model: {selected_name}")

    # ---- Task 7: over-sample the lapsed class in the training rows only
    X_balanced, y_balanced = oversample_lapsed(X_train, y_train)
    print(f"\nBalanced training set: {(y_balanced == 0).sum()} not lapsed, {(y_balanced == 1).sum()} lapsed")

    balanced_model = clone(logistic_search.best_estimator_).fit(X_balanced, y_balanced)
    balanced_scores = show_test_result("Logistic regression trained on balanced data", balanced_model, X_test, y_test)

    print("\nChange after balancing (test set, lapsed class):")
    for metric in ("precision", "recall", "f1"):
        print(f"  {metric:<9} {logistic_scores[metric]:.3f} -> {balanced_scores[metric]:.3f}")

    # ---- Task 8 (a): what drives the predictions
    prep = forest_search.best_estimator_.named_steps["prep"]
    names = prep.get_feature_names_out()
    importances = pd.Series(
        forest_search.best_estimator_.named_steps["model"].feature_importances_, index=names
    )
    print("\nTop 5 random forest features:")
    print(importances.nlargest(5).round(3).to_string())

    logistic_names = logistic_search.best_estimator_.named_steps["prep"].get_feature_names_out()
    coefficients = pd.Series(
        logistic_search.best_estimator_.named_steps["model"].coef_[0], index=logistic_names
    )
    print("\nLogistic regression coefficients (positive = higher lapse risk):")
    for feature in ("num__recency", "num__frequency", "num__cancellation_rate"):
        print(f"  {feature:<26} {coefficients[feature]:+.3f}")

    # ---- Task 8 (b): save, reload, compare
    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "lapse_model.joblib"
    joblib.dump(selected, model_path)
    reloaded = joblib.load(model_path)
    same = np.array_equal(selected.predict(X_test), reloaded.predict(X_test))
    print(f"\nReloaded model gives identical test predictions: {same}")
    assert same, "reloaded model differs from the original"

    # ---- Task 8 (c): retention list for Premium customers in the test set
    segments = pd.read_csv(OUTPUT_DIR / "customer_segments.csv")
    premium_ids = set(segments.loc[segments["is_premium"], "customer_id"])

    test_customers = cohort.loc[X_test.index, "customer_id"]
    is_premium_test = test_customers.isin(premium_ids)

    retention = pd.DataFrame(
        {
            "customer_id": test_customers[is_premium_test],
            "lapse_probability": selected.predict_proba(X_test[is_premium_test])[:, 1].round(4),
            "actually_lapsed": y_test[is_premium_test],
        }
    )
    retention = retention.sort_values("lapse_probability", ascending=False).head(10)
    retention.to_csv(OUTPUT_DIR / "retention_list.csv", index=False)

    print(f"\nRetention list: {len(retention)} Premium customers, "
          f"{int(retention['actually_lapsed'].sum())} of them truly lapsed.")
    print("Note: the Premium flag uses whole-window RFM (September included), "
          "so this list is illustrative, not a leakage-free evaluation.")


if __name__ == "__main__":
    main()
