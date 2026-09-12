import pandas as pd

from src.survival import fit_km_by_segment, logrank_pvalue


def _synthetic_data():
    durations = pd.Series([5, 6, 7, 8, 9] + [50, 55, 60, 65, 70])
    event_observed = pd.Series([1, 1, 1, 1, 1] + [1, 1, 0, 0, 0])
    segment = pd.Series(["short"] * 5 + ["long"] * 5)
    return durations, event_observed, segment


def test_fit_km_by_segment_returns_one_fitter_per_group():
    durations, event_observed, segment = _synthetic_data()
    fitters = fit_km_by_segment(durations, event_observed, segment)
    assert set(fitters.keys()) == {"short", "long"}
    assert fitters["short"].median_survival_time_ < fitters["long"].median_survival_time_


def test_logrank_pvalue_detects_clear_difference():
    durations, event_observed, segment = _synthetic_data()
    p = logrank_pvalue(durations, event_observed, segment, "short", "long")
    assert p < 0.05
