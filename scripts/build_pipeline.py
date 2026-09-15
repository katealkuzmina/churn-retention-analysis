"""Non-interactive driver for the end-to-end churn-retention pipeline.

Validated here as a plain script first (faster iteration on real 30GB+
data than re-running notebook cells), then folded into
churn_retention_analysis.ipynb once every stage is confirmed correct.
"""
from __future__ import annotations

import gc
import time

import duckdb
import numpy as np
import pandas as pd

from src.candidates import scope_to_renewal_candidates
from src.cohorts import build_cohort_retention
from src.economics import (
    breakeven_conversion_rate,
    campaign_expected_profit,
    scale_to_full_population,
)
from src.feature_mart import build_feature_mart
from src.hypothesis_tests import (
    chi_square_independence,
    correct_pvalues,
    two_proportion_ztest,
)
from src.labels import derive_churn_labels
from src.modeling import (
    FEATURE_COLUMNS,
    brier_before_after,
    calibrate_isotonic,
    evaluate,
    train_lightgbm,
    train_logistic_baseline,
)
from src.sampling import stratified_sample
from src.shap_utils import compute_shap_values
from src.survival import logrank_pvalue

CUTOFFS = {
    "train": "2016-11-30",
    "val": "2016-12-31",
    "test": "2017-01-31",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute('SET memory_limit="4GB"')
    con.execute("SET threads TO 4")
    con.execute("SET temp_directory='data/interim/duckdb_tmp'")
    con.execute("CREATE VIEW members AS SELECT * FROM read_parquet('data/interim/members.parquet')")
    con.execute("CREATE VIEW transactions AS SELECT * FROM read_parquet('data/interim/transactions.parquet')")
    con.execute("CREATE VIEW user_logs AS SELECT * FROM read_parquet('data/interim/user_logs.parquet')")
    return con


def main() -> None:
    con = connect()

    log("loading transactions once (reused across cutoffs, labels, cohorts)")
    tx_all = con.execute("SELECT msno, transaction_date, membership_expire_date, is_cancel FROM transactions").df()
    tx_all["transaction_date"] = pd.to_datetime(tx_all["transaction_date"])
    tx_all["membership_expire_date"] = pd.to_datetime(tx_all["membership_expire_date"])
    log(f"  transactions: {len(tx_all):,} rows")

    log("building core feature mart + labels for 3 cutoffs")
    folds = []
    for fold_name, cutoff in CUTOFFS.items():
        t0 = time.time()
        mart = build_feature_mart(con, cutoff)
        labels = derive_churn_labels(tx_all, pd.Timestamp(cutoff), label_horizon_days=30)
        # Candidate scoping: only members whose subscription is actually
        # coming up for renewal shortly after cutoff are "at risk" this
        # period. Without this, the population includes everyone with any
        # transaction history (many mid-subscription, nothing to decide
        # yet), which inflates churn to ~50% instead of the ~6% KKBox's
        # own train.csv reports for the equivalent Jan-cutoff population
        # (validated: this filter reproduces ~956k candidates at ~7.7%
        # churn vs. train.csv's 993k at 6.4%, 95% label agreement on the
        # overlap).
        labels = scope_to_renewal_candidates(labels, pd.Timestamp(cutoff), window_days=28)
        merged = mart.merge(labels[["msno", "is_churn", "expire_at_cutoff"]], on="msno", how="inner")
        merged["fold"] = fold_name
        folds.append(merged)
        log(f"  {fold_name} ({cutoff}): {len(merged):,} rows in {time.time()-t0:.1f}s, "
            f"churn rate {merged['is_churn'].mean():.3%}")

    population = pd.concat(folds, ignore_index=True)
    del folds
    gc.collect()
    log(f"combined population: {len(population):,} rows")

    # Captured here, BEFORE stratified_sample runs below, so downstream
    # economics scaling (scale_to_full_population) reflects the true
    # un-sampled test-fold candidate count, not the ~12%-sampled fold size.
    full_test_population_size = (population["fold"] == "test").sum()
    log(f"full (un-sampled) test-fold population: {full_test_population_size:,} rows")

    log("adding demographic columns")
    extra = con.execute("""
        SELECT
            msno,
            city,
            CASE WHEN bd <= 0 OR bd > 100 THEN NULL ELSE bd END AS bd_cleaned,
            (bd <= 0 OR bd > 100) AS bd_missing,
            gender,
            registered_via
        FROM members
    """).df()
    population = population.merge(extra, on="msno", how="left")

    log("stratified sampling to 300k rows (stratum = has_recent_activity)")
    population["has_recent_activity"] = population["active_days_last_30"] > 0
    sample = stratified_sample(population, "has_recent_activity", sample_size=300_000, random_state=42)
    log(f"sampled population: {len(sample):,} rows")
    sample.to_parquet("data/processed/sampled_population.parquet", index=False)

    log("building cohort retention table (renewal-candidate population)")
    tx_all["month"] = tx_all["transaction_date"].values.astype("datetime64[M]")
    active_periods = tx_all[["msno", "month"]].drop_duplicates().rename(columns={"month": "period"})
    members_df = con.execute("SELECT msno, registration_init_time FROM members").df()
    members_df["registration_init_time"] = pd.to_datetime(members_df["registration_init_time"])
    candidate_msnos = population["msno"].unique()
    members_df = members_df[members_df["msno"].isin(candidate_msnos)]

    data_start = tx_all["transaction_date"].min()
    data_end = tx_all["transaction_date"].max()
    cohort_retention = build_cohort_retention(
        members_df,
        active_periods,
        max_observable_month=lambda cohort_month: (
            (data_end.to_period("M") - cohort_month).n
        ),
        min_observable_month=lambda cohort_month: max(
            0, (data_start.to_period("M") - cohort_month).n
        ),
    )
    cohort_retention.to_parquet("data/processed/cohort_retention.parquet")
    log(f"cohort retention table: {cohort_retention.shape}")

    log("survival analysis (test-fold snapshot)")
    test_fold = sample[sample["fold"] == "test"].merge(
        members_df[["msno", "registration_init_time"]], on="msno", how="left"
    )
    # Duration is time-to-event, tied to the same renewal decision
    # is_churn describes -- not an unrelated calendar-tenure snapshot.
    test_fold["duration_days"] = (
        test_fold["expire_at_cutoff"] - test_fold["registration_init_time"]
    ).dt.days
    survival_results = {}
    for segment_col in ["is_auto_renew", "registered_via"]:
        seg = test_fold[segment_col].astype("string")
        top_two = seg.value_counts().head(2).index.tolist()
        if len(top_two) == 2:
            p = logrank_pvalue(test_fold["duration_days"], test_fold["is_churn"], seg, *top_two)
            survival_results[segment_col] = p
            log(f"  log-rank {segment_col} ({top_two[0]} vs {top_two[1]}): p={p:.4g}")

    log("hypothesis tests with BH correction")
    p_values = []
    test_labels = []
    auto_renew_churn = test_fold.groupby("is_auto_renew")["is_churn"].agg(["sum", "count"])
    if len(auto_renew_churn) == 2:
        successes = tuple(auto_renew_churn["sum"])
        totals = tuple(auto_renew_churn["count"])
        _, p = two_proportion_ztest(successes, totals)
        p_values.append(p)
        test_labels.append("auto_renew_vs_churn")

    reg_via_table = pd.crosstab(test_fold["registered_via"], test_fold["is_churn"])
    if reg_via_table.shape[0] > 1:
        _, p = chi_square_independence(reg_via_table)
        p_values.append(p)
        test_labels.append("registered_via_vs_churn")

    reject_flags = correct_pvalues(p_values) if p_values else []
    for label, p, reject in zip(test_labels, p_values, reject_flags):
        log(f"  {label}: p={p:.4g}, significant_after_BH={reject}")

    log("temporal train/val/test split + LightGBM")
    train_df = sample[sample["fold"] == "train"]
    val_df = sample[sample["fold"] == "val"]
    test_df = sample[sample["fold"] == "test"]
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        log(f"  {name}: {len(df):,} rows, churn rate {df['is_churn'].mean():.3%}")

    model = train_lightgbm(train_df, val_df)
    metrics = evaluate(model, test_df)
    log(f"  test PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")

    log("training logistic-regression baseline for comparison")
    baseline = train_logistic_baseline(train_df)
    baseline_metrics = evaluate(baseline, test_df)
    log(f"  baseline PR-AUC={baseline_metrics['pr_auc']:.4f} ROC-AUC={baseline_metrics['roc_auc']:.4f}")

    log("isotonic calibration")
    calibrated = calibrate_isotonic(model, val_df)
    brier = brier_before_after(model, calibrated, test_df)
    log(f"  Brier before={brier['brier_before']:.4f} after={brier['brier_after']:.4f}")

    log("SHAP")
    shap_values = compute_shap_values(model, test_df[FEATURE_COLUMNS])
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS).sort_values(ascending=False)
    log("  top 5 features by mean |SHAP|:\n" + mean_abs_shap.head(5).to_string())

    log("campaign economics sweep")
    p_churn_raw = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    p_churn_calibrated = calibrated.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    p_series = pd.Series(p_churn_calibrated, index=test_df.index)

    best_k, best_profit = None, -np.inf
    for k in np.arange(0.01, 1.01, 0.01):
        profit = campaign_expected_profit(
            p_series, top_k_fraction=k, conversion_rate=0.15,
            arpu=4.99, avg_lifetime_months=21, contact_cost=3.0,
        )
        if profit > best_profit:
            best_k, best_profit = k, profit
    breakeven = breakeven_conversion_rate(
        p_series, top_k_fraction=best_k, arpu=4.99, avg_lifetime_months=21, contact_cost=3.0,
    )
    log(f"  best top-k={best_k:.0%}, expected profit=${best_profit:,.0f}, breakeven conversion={breakeven:.1%}")

    log("economics scenario table (lifetime/margin sensitivity, scaled to full test population)")
    scenarios = []
    for label, lifetime_months, margin_rate in [
        ("21mo revenue (original assumption)", 21, 1.0),
        ("12mo revenue (target-group-adjusted lifetime)", 12, 1.0),
        ("12mo margin at 40% margin_rate", 12, 0.4),
        ("6mo revenue", 6, 1.0),
    ]:
        sample_profit = campaign_expected_profit(
            p_series, best_k, conversion_rate=0.15, arpu=4.99,
            avg_lifetime_months=lifetime_months, contact_cost=3.0, margin_rate=margin_rate,
        )
        full_profit = scale_to_full_population(
            sample_profit, sample_size=len(p_series), full_population_size=full_test_population_size,
        )
        scenarios.append({"scenario": label, "sample_profit": sample_profit, "full_population_profit": full_profit})

    scenario_table = pd.DataFrame(scenarios)
    log("\n" + scenario_table.to_string(index=False))

    log("writing scored feature mart")
    scored = test_df.copy()
    scored["p_churn_raw"] = p_churn_raw
    scored["p_churn_calibrated"] = p_churn_calibrated
    scored.to_parquet("data/processed/scored_feature_mart.parquet", index=False)
    log(f"  wrote {len(scored):,} rows to data/processed/scored_feature_mart.parquet")

    log("done")


if __name__ == "__main__":
    main()
