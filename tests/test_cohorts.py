import pandas as pd

from src.cohorts import build_cohort_retention


def test_build_cohort_retention_computes_expected_rates():
    members = pd.DataFrame({
        "msno": ["A", "B"],
        "registration_init_time": pd.to_datetime(["2016-01-15", "2016-01-20"]),
    })
    active_periods = pd.DataFrame({
        "msno": ["A", "A", "B"],
        "period": pd.to_datetime(["2016-01-01", "2016-02-01", "2016-01-01"]),
    })

    retention = build_cohort_retention(members, active_periods)

    assert retention.loc[pd.Period("2016-01", "M"), 0] == 1.0
    assert retention.loc[pd.Period("2016-01", "M"), 1] == 0.5
