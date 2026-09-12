import pandas as pd
import pytest

from src.economics import (
    breakeven_conversion_rate,
    campaign_expected_profit,
    ev_per_contact,
    ltv,
)


def test_ltv_is_arpu_times_lifetime():
    assert ltv(arpu=4.99, avg_lifetime_months=21) == pytest.approx(104.79)


def test_ev_per_contact_matches_hand_calculation():
    result = ev_per_contact(
        p_churn=0.5, conversion_rate=0.2, arpu=4.99,
        avg_lifetime_months=21, contact_cost=3.0,
    )
    expected = 0.5 * 0.2 * (4.99 * 21) - 3.0
    assert result == pytest.approx(expected)


def test_campaign_expected_profit_sums_over_top_k():
    p_churn = pd.Series([0.9, 0.8, 0.1, 0.05])
    profit = campaign_expected_profit(
        p_churn, top_k_fraction=0.5, conversion_rate=0.3,
        arpu=4.99, avg_lifetime_months=21, contact_cost=3.0,
    )
    expected = sum(p * 0.3 * (4.99 * 21) - 3.0 for p in [0.9, 0.8])
    assert profit == pytest.approx(expected)


def test_breakeven_conversion_rate_zeroes_out_profit():
    p_churn = pd.Series([0.9, 0.8, 0.1, 0.05])
    rate = breakeven_conversion_rate(
        p_churn, top_k_fraction=0.5, arpu=4.99,
        avg_lifetime_months=21, contact_cost=3.0,
    )
    profit_at_breakeven = campaign_expected_profit(
        p_churn, top_k_fraction=0.5, conversion_rate=rate,
        arpu=4.99, avg_lifetime_months=21, contact_cost=3.0,
    )
    assert profit_at_breakeven == pytest.approx(0.0, abs=1e-9)
