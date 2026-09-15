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
