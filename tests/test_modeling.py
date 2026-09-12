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
