import pandas as pd


def build_cohort_retention(
    members: pd.DataFrame,
    active_periods: pd.DataFrame,
    max_observable_month,
    min_observable_month,
) -> pd.DataFrame:
    """Monthly cohort retention table.

    members: columns [msno, registration_init_time] -- must already be
    scoped to the population this retention curve is about (e.g. the
    renewal-candidate population), NOT every registrant in the raw
    members table -- otherwise the denominator includes members who
    never had a subscription to retain in the first place, and month-0
    retention reads far below 100%.

    active_periods: columns [msno, period] -- one row per (msno, month)
    the member held an active subscription.

    max_observable_month: callable, cohort_month (a pandas Period) ->
    int, the last months_since_signup value the data can actually speak
    to for that cohort (e.g. based on when the source data ends). Cells
    beyond this are NaN (unobserved), not 0 (observed zero retention) --
    the data hasn't caught up to that month yet (right-censoring).

    min_observable_month: callable, cohort_month (a pandas Period) -> int,
    the first months_since_signup value the data can actually speak to for
    that cohort (e.g. based on when the source data begins). Cells before
    this are also NaN, not 0 -- for cohorts that registered before the
    data's coverage window starts, those early calendar months simply
    predate the data entirely (left-censoring/left-truncation), so there
    was never a chance to observe them, let alone observe zero retention.

    Together the two bounds mark exactly the [min, max] months_since_signup
    range the data can speak to for each cohort; only NaN cells inside that
    range get zero-filled (genuinely-observed absence). NaN cells outside
    it -- whether too early (before data existed) or too late (before data
    caught up) -- stay NaN.
    """
    cohort_month = members.set_index("msno")["registration_init_time"].dt.to_period("M")

    activity = active_periods.copy()
    activity["cohort_month"] = activity["msno"].map(cohort_month)
    activity["period_month"] = activity["period"].dt.to_period("M")
    activity["months_since_signup"] = (
        activity["period_month"].astype(int) - activity["cohort_month"].astype(int)
    )
    activity = activity[activity["months_since_signup"] >= 0]

    cohort_sizes = members.assign(cohort_month=cohort_month.values).groupby(
        "cohort_month"
    )["msno"].nunique()

    counts = (
        activity.groupby(["cohort_month", "months_since_signup"])["msno"]
        .nunique()
        .unstack()
    )

    for cohort_month_value in counts.index:
        max_month = max_observable_month(cohort_month_value)
        min_month = min_observable_month(cohort_month_value)
        for month_col in counts.columns:
            if (
                min_month <= month_col <= max_month
                and pd.isna(counts.loc[cohort_month_value, month_col])
            ):
                counts.loc[cohort_month_value, month_col] = 0

    return counts.div(cohort_sizes, axis=0)
