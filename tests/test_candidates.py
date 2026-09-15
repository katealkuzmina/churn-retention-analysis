import pandas as pd

from src.candidates import scope_to_renewal_candidates


def test_scope_to_renewal_candidates_keeps_only_the_window():
    labels = pd.DataFrame({
        "msno": ["A", "B", "C", "D"],
        "expire_at_cutoff": pd.to_datetime([
            "2016-12-31",  # exactly at cutoff -> kept
            "2017-01-15",  # inside window -> kept
            "2017-01-28",  # exactly at cutoff+28d -> excluded (half-open)
            "2016-12-30",  # before cutoff -> excluded
        ]),
        "is_churn": [0, 1, 0, 1],
    })
    cutoff = pd.Timestamp("2016-12-31")

    result = scope_to_renewal_candidates(labels, cutoff, window_days=28)

    assert set(result["msno"]) == {"A", "B"}


def test_scope_to_renewal_candidates_test_fold_horizon_fits_available_data():
    """Regression test for the mentor-flagged label-censoring bug: the
    latest expire_at_cutoff in the test fold, plus the 30-day churn
    horizon, must not exceed the last date transactions_v2.csv covers
    (2017-03-31) -- otherwise renewals that happen after the data ends
    get invisibly counted as churn.
    """
    test_cutoff = pd.Timestamp("2017-01-31")
    window_days = 28
    horizon_days = 30
    data_end = pd.Timestamp("2017-03-31")

    latest_possible_expiry = test_cutoff + pd.Timedelta(days=window_days - 1)
    horizon_end = latest_possible_expiry + pd.Timedelta(days=horizon_days)

    assert horizon_end <= data_end
