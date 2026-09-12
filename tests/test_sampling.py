import pandas as pd

from src.sampling import stratified_sample


def test_stratified_sample_preserves_group_ratio():
    df = pd.DataFrame({
        "msno": [f"U{i}" for i in range(1000)],
        "has_recent_activity": [True] * 800 + [False] * 200,
    })
    sample = stratified_sample(df, "has_recent_activity", sample_size=100, random_state=1)

    assert len(sample) == 100
    active_share = sample["has_recent_activity"].mean()
    assert 0.72 <= active_share <= 0.88  # target 0.80, rounding slack
