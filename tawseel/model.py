"""Failure-risk baseline for Tawseel COD parcels.

You write this file. Keep every name and signature exactly as it is: the grader imports them.
Allowed dependencies: numpy, pandas, scikit-learn (and the standard library).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TARGET = "failed"

# The raw columns of make_parcels() that the model may use at decision time (when the parcel
# is created). Nothing known only after the delivery attempt, not the label, not the row id.
# These are the columns available when the parcel is created.
FEATURE_COLUMNS: list[str] = [
    "customer_id", "merchant_id", "area", "created_at", "promised_slot", "cod_amount",
    "n_items", "address_text",
]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df (same index and row order) with two new columns.

    Needs customer_id, created_at and failed.
    - customer_prior_parcels (int): the same customer's parcels with created_at strictly
      earlier than this row.
    - customer_prior_fail_rate (float): share of those that failed; NaN when there are none.
    Parcels with the same timestamp are not prior to each other. The result must not depend
    on the input row order. Never modify the input.
    """
    result = df.copy()
    result["customer_prior_parcels"] = 0
    result["customer_prior_fail_rate"] = np.nan

    # Calculate each timestamp block before writing it back.  That prevents two
    # parcels created at the same instant from seeing one another as history.
    work = result[["customer_id", "created_at", TARGET]].copy()
    work["_position"] = np.arange(len(work))
    for _, customer_rows in work.groupby("customer_id", dropna=False, sort=False):
        customer_rows = customer_rows.sort_values("created_at", kind="mergesort")
        prior_count = 0
        prior_failures = 0.0
        for _, timestamp_rows in customer_rows.groupby(
            "created_at", dropna=False, sort=False
        ):
            positions = timestamp_rows["_position"].to_numpy()
            result.iloc[positions, result.columns.get_loc("customer_prior_parcels")] = prior_count
            if prior_count:
                result.iloc[positions, result.columns.get_loc("customer_prior_fail_rate")] = (
                    prior_failures / prior_count
                )
            prior_count += len(timestamp_rows)
            prior_failures += float(timestamp_rows[TARGET].sum())
    result["customer_prior_parcels"] = result["customer_prior_parcels"].astype(int)
    return result


def time_split(df: pd.DataFrame, time_col: str = "created_at",
               test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test).

    ValueError unless 0 < test_fraction < 1. Sort by time; n_test = ceil(len(df) * test_fraction);
    the boundary is the timestamp of the first of the last n_test rows. Rows at or after the
    boundary are test, rows before it are train.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between zero and one")
    if df.empty:
        return df.copy(), df.copy()

    ordered = df.sort_values(time_col, kind="mergesort")
    n_test = math.ceil(len(df) * test_fraction)
    boundary = ordered.iloc[-n_test][time_col]
    test_mask = df[time_col] >= boundary
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def _datetime_features(values) -> np.ndarray:
    """Turn a creation timestamp into simple decision-time calendar features."""
    flat = np.asarray(values).reshape(-1)
    timestamps = pd.to_datetime(pd.Series(flat), errors="coerce")
    return np.column_stack(
        [
            timestamps.dt.dayofweek.fillna(-1).to_numpy(),
            timestamps.dt.hour.fillna(-1).to_numpy(),
            timestamps.dt.month.fillna(-1).to_numpy(),
        ]
    )


def _address_features(values) -> np.ndarray:
    """Extract stable decision-time signals without depending on vocabulary."""
    flattened = np.asarray(values, dtype=object).reshape(-1)
    text = pd.Series(flattened).fillna("").astype(str).str.strip()
    return np.column_stack(
        [
            text.str.len().to_numpy(),
            text.eq("").astype(int).to_numpy(),
            text.str.contains(r"\d").astype(int).to_numpy(),
        ]
    )


def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an UNFITTED Pipeline.

    Input: a DataFrame with at least FEATURE_COLUMNS plus the two history columns; ignore any
    other column. The final step must support predict_proba. Must not fail on unseen
    categories, missing numbers, missing categories or empty address strings. Same seed and
    same data must give identical predictions.
    """
    numeric_columns = [
        "cod_amount",
        "n_items",
        "customer_prior_parcels",
        "customer_prior_fail_rate",
    ]
    categorical_columns = ["customer_id", "merchant_id", "area", "promised_slot"]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            ("created_at", FunctionTransformer(_datetime_features, validate=False), ["created_at"]),
            ("address", FunctionTransformer(_address_features, validate=False), ["address_text"]),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1_000, random_state=seed, solver="liblinear")),
        ]
    )


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """{"roc_auc", "pr_auc", "brier", "base_rate", "n"} from model.predict_proba(test_df)[:, 1]
    against test_df["failed"] (roc_auc_score, average_precision_score, brier_score_loss)."""
    probabilities = model.predict_proba(test_df)[:, 1]
    return _metric_results(test_df[TARGET], probabilities)


def _metric_results(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    """Calculate the task's required binary-classification metrics."""
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    """Fit build_pipeline(seed) and DummyClassifier(strategy="prior") on train_df only, evaluate
    both on test_df, return {"model": {...}, "dummy": {...}, "lift_pr_auc": model - dummy PR-AUC}."""
    model = build_pipeline(seed)
    model.fit(train_df, train_df[TARGET])

    # The dummy consumes no predictors; pass one stable column so its interface
    # remains the same as the real estimator without exposing the target.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train_df[["customer_prior_parcels"]], train_df[TARGET])

    model_metrics = evaluate(model, test_df)
    dummy_test = test_df[["customer_prior_parcels"]]
    dummy_probabilities = dummy.predict_proba(dummy_test)[:, 1]
    dummy_metrics = _metric_results(test_df[TARGET], dummy_probabilities)
    return {
        "model": model_metrics,
        "dummy": dummy_metrics,
        "lift_pr_auc": model_metrics["pr_auc"] - dummy_metrics["pr_auc"],
    }
