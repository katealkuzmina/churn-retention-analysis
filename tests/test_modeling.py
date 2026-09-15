import numpy as np
import pandas as pd

from src.modeling import (
    FEATURE_COLUMNS,
    brier_before_after,
    calibrate_isotonic,
    evaluate,
    train_lightgbm,
)


def _synthetic_split(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    logits = -2.0 * df["activity_trend_30d"] + rng.normal(scale=0.5, size=n)
    df["is_churn"] = (logits > np.quantile(logits, 0.85)).astype(int)
    third = n // 3
    return df.iloc[:third], df.iloc[third:2 * third], df.iloc[2 * third:]


def test_train_lightgbm_beats_random_on_synthetic_signal():
    train_df, val_df, test_df = _synthetic_split()
    model = train_lightgbm(train_df, val_df)
    metrics = evaluate(model, test_df)
    assert metrics["roc_auc"] > 0.7


def test_calibration_improves_or_maintains_brier_score():
    train_df, val_df, test_df = _synthetic_split()
    model = train_lightgbm(train_df, val_df)
    calibrated = calibrate_isotonic(model, val_df)
    scores = brier_before_after(model, calibrated, test_df)
    assert scores["brier_after"] <= scores["brier_before"] + 0.01


def test_train_lightgbm_early_stops_on_average_precision_not_logloss():
    """Regression test: without first_metric_only=True, LightGBM's
    early stopping waits for BOTH the default binary_logloss metric and
    average_precision to stop improving, and binary_logloss (which
    converges in a couple of rounds on this kind of imbalanced data)
    ends up gating best_iteration_ -- silently truncating the model to
    a handful of trees regardless of what average_precision does later.
    first_metric_only=True fixes it to actually watch average_precision.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    y = pd.Series((X["f0"] + X["f1"] * 0.5 + rng.normal(scale=0.5, size=n) > 1.5).astype(int))
    train_df = pd.concat([X.iloc[:1000], y.iloc[:1000].rename("is_churn")], axis=1)
    val_df = pd.concat([X.iloc[1000:], y.iloc[1000:].rename("is_churn")], axis=1)

    model = train_lightgbm(train_df, val_df, feature_columns=list(X.columns))

    # With a real, still-improving-on-AP signal like this, stopping at
    # a handful of trees means logloss (not AP) gated the stop.
    assert model.best_iteration_ > 10
