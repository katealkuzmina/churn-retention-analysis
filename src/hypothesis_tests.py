import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest


def two_proportion_ztest(
    successes: tuple[int, int], totals: tuple[int, int]
) -> tuple[float, float]:
    stat, p_value = proportions_ztest(count=list(successes), nobs=list(totals))
    return stat, p_value


def chi_square_independence(contingency_table: pd.DataFrame) -> tuple[float, float]:
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table.values)
    return chi2_stat, p_value


def correct_pvalues(p_values: list[float], method: str = "fdr_bh") -> list[bool]:
    reject, _, _, _ = multipletests(p_values, alpha=0.05, method=method)
    return [bool(r) for r in reject]
