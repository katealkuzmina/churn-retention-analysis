import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def fit_km_by_segment(
    durations: pd.Series,
    event_observed: pd.Series,
    segment: pd.Series,
) -> dict:
    fitters = {}
    for value in segment.dropna().unique():
        mask = segment == value
        kmf = KaplanMeierFitter(label=str(value))
        kmf.fit(durations[mask], event_observed=event_observed[mask])
        fitters[value] = kmf
    return fitters


def logrank_pvalue(
    durations: pd.Series,
    event_observed: pd.Series,
    segment: pd.Series,
    group_a,
    group_b,
) -> float:
    mask_a = segment == group_a
    mask_b = segment == group_b
    result = logrank_test(
        durations[mask_a], durations[mask_b],
        event_observed_A=event_observed[mask_a],
        event_observed_B=event_observed[mask_b],
    )
    return result.p_value
