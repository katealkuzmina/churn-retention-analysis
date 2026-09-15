import pandas as pd


def derive_churn_labels(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    label_horizon_days: int = 30,
) -> pd.DataFrame:
    """One row per msno with a membership on file as of cutoff_date.

    `expire_at_cutoff` is membership_expire_date from the last transaction
    strictly before cutoff_date. is_churn=1 unless a genuine renewal
    transaction exists: dated on/after cutoff_date (so it's actually new
    information, not a data-ordering artifact from before the cutoff),
    dated before expire_at_cutoff + label_horizon_days (the observation
    window), extending membership_expire_date past expire_at_cutoff, and
    not itself a cancellation (is_cancel=0) -- a cancellation transaction
    can carry a membership_expire_date past the old expiry (e.g. "cancel
    effective at end of current paid period") without being a renewal.

    Requires an `is_cancel` column on `transactions` (0/1).
    """
    before_cutoff = transactions[transactions["transaction_date"] < cutoff_date]
    last_before = (
        before_cutoff.sort_values(["transaction_date", "membership_expire_date"])
        .groupby("msno")
        .tail(1)
        .set_index("msno")["membership_expire_date"]
        .rename("expire_at_cutoff")
    )

    merged = transactions.merge(last_before, on="msno", how="inner")
    is_renewal = (
        (merged["transaction_date"] >= cutoff_date)
        & (merged["transaction_date"] < merged["expire_at_cutoff"] + pd.Timedelta(days=label_horizon_days))
        & (merged["membership_expire_date"] > merged["expire_at_cutoff"])
        & (merged["is_cancel"] == 0)
    )
    renewed_msnos = set(merged.loc[is_renewal, "msno"])

    labels = last_before.reset_index()
    labels["is_churn"] = (~labels["msno"].isin(renewed_msnos)).astype(int)
    return labels[["msno", "expire_at_cutoff", "is_churn"]]
