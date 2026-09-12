from __future__ import annotations

import pandas as pd


def ltv(arpu: float, avg_lifetime_months: float) -> float:
    return arpu * avg_lifetime_months


def ev_per_contact(
    p_churn: float,
    conversion_rate: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
) -> float:
    """Expected profit from contacting one member with churn probability
    p_churn, given the retention campaign's conversion_rate.
    """
    return p_churn * conversion_rate * ltv(arpu, avg_lifetime_months) - contact_cost


def campaign_expected_profit(
    p_churn: pd.Series,
    top_k_fraction: float,
    conversion_rate: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
) -> float:
    """Total expected profit from contacting the top top_k_fraction of
    p_churn (highest risk first)."""
    n_contacted = max(1, round(len(p_churn) * top_k_fraction))
    contacted = p_churn.sort_values(ascending=False).head(n_contacted)
    return sum(
        ev_per_contact(p, conversion_rate, arpu, avg_lifetime_months, contact_cost)
        for p in contacted
    )


def breakeven_conversion_rate(
    p_churn: pd.Series,
    top_k_fraction: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
) -> float:
    """conversion_rate at which campaign_expected_profit(...) == 0 for the
    given top_k_fraction: contact_cost / (mean(p_churn over contacted) * ltv)."""
    n_contacted = max(1, round(len(p_churn) * top_k_fraction))
    contacted = p_churn.sort_values(ascending=False).head(n_contacted)
    mean_p_churn = contacted.mean()
    return contact_cost / (mean_p_churn * ltv(arpu, avg_lifetime_months))
