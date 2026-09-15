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

    retention = build_cohort_retention(
        members,
        active_periods,
        max_observable_month=lambda cohort: 999,
        min_observable_month=lambda cohort: 0,
    )

    assert retention.loc[pd.Period("2016-01", "M"), 0] == 1.0
    assert retention.loc[pd.Period("2016-01", "M"), 1] == 0.5


def test_build_cohort_retention_uses_nan_not_zero_for_months_with_no_data():
    """A cohort with no observed activity rows in a given month (because
    the data simply doesn't extend that far, not because everyone
    churned) must show NaN there, not 0 -- 0 reads as 'nobody retained',
    which conflates 'no data' with 'no survivors'.
    """
    members = pd.DataFrame({
        "msno": ["A", "B"],
        "registration_init_time": pd.to_datetime(["2016-01-01", "2016-06-01"]),
    })
    # A is active in months 0 and 1 after signup; B has no rows at all
    # past its signup month within the data's range (data simply ends).
    active_periods = pd.DataFrame({
        "msno": ["A", "A", "B"],
        "period": pd.to_datetime(["2016-01-15", "2016-02-15", "2016-06-15"]),
    })

    # The data's activity rows end at 2016-06 (B's own signup month), so
    # a cohort's observable horizon is (data_end - cohort_month), mirroring
    # the real caller in scripts/build_pipeline.py. B's cohort therefore has
    # a horizon of 0 months -- month 1 is genuinely beyond the data's reach.
    data_end = pd.Period("2016-06", "M")
    retention = build_cohort_retention(
        members,
        active_periods,
        max_observable_month=lambda cohort: (data_end - cohort).n,
        min_observable_month=lambda cohort: 0,
    )

    # B's cohort (2016-06) has no rows for months_since_signup=1: that
    # cell must be NaN (unobserved), not 0.0 (observed zero retention).
    assert pd.isna(retention.loc[pd.Period("2016-06", "M"), 1])


def test_build_cohort_retention_uses_nan_not_zero_for_months_before_data_coverage_starts():
    """A cohort registered before the data's coverage window even begins
    (left-censoring/left-truncation) must show NaN for the early
    months_since_signup values that predate the data, not 0.0 -- the data
    never existed for that period, so it's unobserved, not observed-zero.
    """
    members = pd.DataFrame({
        "msno": ["A", "B", "C"],
        # A signed up before the data's coverage window starts (2016-03);
        # B signed up right at the start of the window; C one month before
        # the window (used only to give months_since_signup=1 a column in
        # the pivot table via C's own, in-range, cohort).
        "registration_init_time": pd.to_datetime(["2016-01-01", "2016-03-01", "2016-02-01"]),
    })
    # active_periods only has rows from 2016-03 onward (the data's actual
    # coverage window). A has a row once the window opens (months_since_signup
    # = 2, since A's cohort is 2016-01); nothing for A's own months 0 or 1,
    # because the data simply didn't exist yet, not because A churned.
    active_periods = pd.DataFrame({
        "msno": ["A", "B", "C"],
        "period": pd.to_datetime(["2016-03-15", "2016-03-15", "2016-03-15"]),
    })

    data_start = pd.Period("2016-03", "M")
    retention = build_cohort_retention(
        members,
        active_periods,
        max_observable_month=lambda cohort: 999,
        min_observable_month=lambda cohort: max(0, (data_start - cohort).n),
    )

    # A's cohort (2016-01) has no rows for months_since_signup 0 or 1 --
    # those calendar months (2016-01, 2016-02) predate the data's coverage
    # window, so they must be NaN, not 0.0.
    assert pd.isna(retention.loc[pd.Period("2016-01", "M"), 0])
    assert pd.isna(retention.loc[pd.Period("2016-01", "M"), 1])
    # months_since_signup=2 (calendar 2016-03) is within the data's coverage
    # window and A has a real row there, so it must reflect actual data.
    assert retention.loc[pd.Period("2016-01", "M"), 2] == 1.0
