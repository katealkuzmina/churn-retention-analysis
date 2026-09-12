import pandas as pd


def build_cohort_retention(
    members: pd.DataFrame,
    active_periods: pd.DataFrame,
) -> pd.DataFrame:
    """Monthly cohort retention table.

    members: columns [msno, registration_init_time].
    active_periods: columns [msno, period] -- one row per (msno, month)
    the member held an active subscription.
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
        .unstack(fill_value=0)
    )
    return counts.div(cohort_sizes, axis=0)
