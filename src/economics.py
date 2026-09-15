from __future__ import annotations

import pandas as pd


def ltv(arpu: float, avg_lifetime_months: float, margin_rate: float = 1.0) -> float:
    """Lifetime value in dollars of MARGIN, not revenue, once margin_rate
    is set below 1.0 -- the campaign spends contact_cost against margin
    it actually keeps, not gross revenue.
    """
    return arpu * avg_lifetime_months * margin_rate


def ev_per_contact(
    p_churn: float,
    conversion_rate: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
    margin_rate: float = 1.0,
) -> float:
    """Expected profit from contacting one member with churn probability
    p_churn, given the retention campaign's conversion_rate.
    """
    return p_churn * conversion_rate * ltv(arpu, avg_lifetime_months, margin_rate) - contact_cost


def campaign_expected_profit(
    p_churn: pd.Series,
    top_k_fraction: float,
    conversion_rate: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
    margin_rate: float = 1.0,
) -> float:
    """Total expected profit from contacting the top top_k_fraction of
    p_churn (highest risk first)."""
    n_contacted = max(1, round(len(p_churn) * top_k_fraction))
    contacted = p_churn.sort_values(ascending=False).head(n_contacted)
    return sum(
        ev_per_contact(p, conversion_rate, arpu, avg_lifetime_months, contact_cost, margin_rate)
        for p in contacted
    )


def breakeven_conversion_rate(
    p_churn: pd.Series,
    top_k_fraction: float,
    arpu: float,
    avg_lifetime_months: float,
    contact_cost: float,
    margin_rate: float = 1.0,
) -> float:
    """conversion_rate at which campaign_expected_profit(...) == 0 for the
    given top_k_fraction: contact_cost / (mean(p_churn over contacted) * ltv)."""
    n_contacted = max(1, round(len(p_churn) * top_k_fraction))
    contacted = p_churn.sort_values(ascending=False).head(n_contacted)
    mean_p_churn = contacted.mean()
    return contact_cost / (mean_p_churn * ltv(arpu, avg_lifetime_months, margin_rate))


def scale_to_full_population(sample_profit: float, sample_size: int, full_population_size: int) -> float:
    """Scales a per-contact-summed profit figure computed on a sample up
    to what contacting the equivalent top-k% of the FULL population
    would yield -- campaign_expected_profit only sums over whatever
    Series it's handed, so a profit number computed on a 12%-ish sample
    understates the real campaign's total profit by roughly that same
    factor unless the caller scales it back up explicitly.
    """
    return sample_profit * (full_population_size / sample_size)
