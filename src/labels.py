import pandas as pd


def derive_churn_labels(
    transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    label_horizon_days: int = 30,
) -> pd.DataFrame:
    """One row per msno with a membership on file as of cutoff_date.

    `expire_at_cutoff` is membership_expire_date from the last transaction
    strictly before cutoff_date. is_churn=1 unless a transaction exists
    that extends membership_expire_date past expire_at_cutoff, dated
    before expire_at_cutoff + label_horizon_days.

    No lower bound on transaction_date: per KKBox's own reference
    labeller (WSDMChurnLabeller.scala, shipped with the competition
    data), the renewal gap is transaction_date - expire_at_cutoff and is
    routinely *negative* -- auto-renewal charges typically land on or
    before the old expiry, not after it. Requiring transaction_date >
    expire_at_cutoff (an earlier version of this function did) misreads
    the normal early-renewal case as churn.
    """
    before_cutoff = transactions[transactions["transaction_date"] < cutoff_date]
    last_before = (
        before_cutoff.sort_values("transaction_date")
        .groupby("msno")
        .tail(1)
        .set_index("msno")["membership_expire_date"]
        .rename("expire_at_cutoff")
    )

    merged = transactions.merge(last_before, on="msno", how="inner")
    is_renewal = (
        (merged["transaction_date"] < merged["expire_at_cutoff"] + pd.Timedelta(days=label_horizon_days))
        & (merged["membership_expire_date"] > merged["expire_at_cutoff"])
    )
    renewed_msnos = set(merged.loc[is_renewal, "msno"])

    labels = last_before.reset_index()
    labels["is_churn"] = (~labels["msno"].isin(renewed_msnos)).astype(int)
    return labels[["msno", "expire_at_cutoff", "is_churn"]]
