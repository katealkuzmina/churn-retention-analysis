from __future__ import annotations

import lightgbm as lgb
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FEATURE_COLUMNS = [
    "tenure_days", "is_auto_renew", "payment_method_id", "plan_list_price",
    "actual_amount_paid", "discount_flag", "num_transactions_last_90d",
    "num_cancels_lifetime", "active_days_last_30", "total_secs_last_30",
    "total_secs_prior_30", "activity_trend_30d", "days_since_last_log",
]


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str = "is_churn",
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> lgb.LGBMClassifier:
    """Trains with early stopping on the validation fold. The caller is
    responsible for splitting train_df/val_df along non-overlapping
    cutoffs (spec Sec 3) -- this function only fits.
    """
    pos_rate = train_df[label_col].mean()
    scale_pos_weight = (1 - pos_rate) / pos_rate

    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=42,
    )
    model.fit(
        train_df[feature_columns], train_df[label_col],
        eval_set=[(val_df[feature_columns], val_df[label_col])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def evaluate(
    model, test_df: pd.DataFrame, label_col: str = "is_churn",
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict[str, float]:
    proba = model.predict_proba(test_df[feature_columns])[:, 1]
    return {
        "pr_auc": average_precision_score(test_df[label_col], proba),
        "roc_auc": roc_auc_score(test_df[label_col], proba),
    }


def calibrate_isotonic(
    model, val_df: pd.DataFrame, label_col: str = "is_churn",
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> CalibratedClassifierCV:
    """Fits isotonic calibration on top of an already-fitted model.

    sklearn >=1.6 dropped cv="prefit" -- FrozenEstimator is the current
    way to tell CalibratedClassifierCV "this estimator is already fit,
    calibrate it as-is against val_df" rather than refitting internally.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(val_df[feature_columns], val_df[label_col])
    return calibrated


def brier_before_after(
    model, calibrated, test_df: pd.DataFrame, label_col: str = "is_churn",
    feature_columns: list[str] = FEATURE_COLUMNS,
) -> dict[str, float]:
    raw_proba = model.predict_proba(test_df[feature_columns])[:, 1]
    calibrated_proba = calibrated.predict_proba(test_df[feature_columns])[:, 1]
    return {
        "brier_before": brier_score_loss(test_df[label_col], raw_proba),
        "brier_after": brier_score_loss(test_df[label_col], calibrated_proba),
    }
