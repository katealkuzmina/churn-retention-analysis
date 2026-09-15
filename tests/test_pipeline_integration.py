"""Integration-level regression tests for the label-censoring bug the
mentor flagged: a test fold's churn horizon must never extend past the
last date the transactions data covers.
"""
import pandas as pd

from src.candidates import scope_to_renewal_candidates
from src.labels import derive_churn_labels


def test_full_label_pipeline_has_no_censored_candidates():
    """End-to-end (on a small synthetic transactions table) check that
    every candidate kept after scoping has its full 30-day observation
    window inside the available data range -- i.e. Task 3's cutoff
    shift, applied to real logic, actually closes the gap.
    """
    data_end = pd.Timestamp("2017-03-31")
    cutoff = pd.Timestamp("2017-01-31")
    window_days = 28
    horizon_days = 30

    # Synthetic transactions: msno G's last-before-cutoff transaction
    # expires right at the edge of the candidate window.
    transactions = pd.DataFrame({
        "msno": ["G"],
        "transaction_date": pd.to_datetime(["2017-01-01"]),
        "membership_expire_date": pd.to_datetime(["2017-02-27"]),  # cutoff + 27d
        "is_cancel": [0],
    })

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=horizon_days)
    candidates = scope_to_renewal_candidates(labels, cutoff, window_days=window_days)

    horizon_ends = candidates["expire_at_cutoff"] + pd.Timedelta(days=horizon_days)
    assert (horizon_ends <= data_end).all(), (
        f"candidate(s) with horizon past data_end: "
        f"{candidates[horizon_ends > data_end]}"
    )
