import numpy as np
import pandas as pd


def stratified_sample(
    feature_mart: pd.DataFrame,
    strata_col: str,
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample sample_size rows, preserving each stratum's population share."""
    if sample_size >= len(feature_mart):
        return feature_mart.copy()

    rng = np.random.default_rng(random_state)
    parts = []
    for _, group in feature_mart.groupby(strata_col):
        n = max(1, round(sample_size * len(group) / len(feature_mart)))
        n = min(n, len(group))
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(group.loc[idx])
    return pd.concat(parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
