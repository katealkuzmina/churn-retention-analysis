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


def test_logrank_pvalue_is_sensitive_to_which_column_is_duration():
    """Regression test for the mentor-flagged bug: duration must be
    causally tied to the event, not an unrelated calendar-tenure value.
    Using a duration column that's independent of the event (pure noise
    relative to it) should NOT produce the same low p-value you'd get
    from a duration column that's actually correlated with the event.
    """
    n = 200
    segment = pd.Series((["a"] * (n // 2)) + (["b"] * (n // 2)))
    event = pd.Series(([1] * (n // 2)) + ([0] * (n // 2)))  # perfectly split by segment

    # duration correlated with the event-carrying segment split
    causal_duration = pd.Series(([10] * (n // 2)) + ([100] * (n // 2)), dtype=float)
    p_causal = logrank_pvalue(causal_duration, event, segment, "a", "b")

    # duration with no relationship to segment/event at all
    import numpy as np
    rng = np.random.default_rng(0)
    unrelated_duration = pd.Series(rng.normal(loc=50, scale=1, size=n))
    p_unrelated = logrank_pvalue(unrelated_duration, event, segment, "a", "b")

    assert p_causal < 0.01
    assert p_unrelated > p_causal
