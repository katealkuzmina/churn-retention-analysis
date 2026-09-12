import pandas as pd

from src.hypothesis_tests import (
    chi_square_independence,
    correct_pvalues,
    two_proportion_ztest,
)


def test_two_proportion_ztest_detects_clear_difference():
    _, p_value = two_proportion_ztest(successes=(50, 500), totals=(1000, 1000))
    assert p_value < 0.01


def test_chi_square_independence_on_associated_table():
    table = pd.DataFrame(
        {"churned": [80, 20], "retained": [20, 880]}, index=["group_a", "group_b"]
    )
    _, p_value = chi_square_independence(table)
    assert p_value < 0.01


def test_correct_pvalues_controls_false_discoveries():
    p_values = [0.001, 0.01, 0.2, 0.5, 0.9]
    reject = correct_pvalues(p_values)
    assert reject[0] is True
    assert reject[-1] is False
    assert len(reject) == len(p_values)
