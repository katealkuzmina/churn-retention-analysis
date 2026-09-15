import pandas as pd


def scope_to_renewal_candidates(
    labels: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    window_days: int = 28,
) -> pd.DataFrame:
    """Restricts derive_churn_labels() output to members actually up for
    renewal shortly after cutoff_date.

    Without this, the population is everyone with any transaction
    history -- most of them mid-subscription with nothing to decide yet
    -- which inflates apparent churn to ~50%. Window is half-open:
    [cutoff_date, cutoff_date + window_days).
    """
    return labels[
        (labels["expire_at_cutoff"] >= cutoff_date)
        & (labels["expire_at_cutoff"] < cutoff_date + pd.Timedelta(days=window_days))
    ]
