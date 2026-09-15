"""One-off generator for churn_retention_analysis.ipynb via nbformat.

Not a project deliverable itself -- run once to (re)build the notebook
file, which is what actually gets executed and committed.

STALE / NOT CURRENTLY MAINTAINED: this generator predates most of the
15-task mentor-review remediation (old CUTOFFS, missing `is_cancel` in the
tx_all query, inline candidate filtering instead of `src/candidates.py`,
etc.). Do not run it -- it would overwrite the current, correct,
committed notebook with a broken pre-remediation version. Needs a full
sync to the current `src/` and `scripts/build_pipeline.py` before it's
safe to run again (tracked as a follow-up task).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Subscription Churn: From Prediction to Retention Economics

KKBox churn prediction + retention economics. This notebook builds the
SQL feature mart, runs cohort/survival analysis and hypothesis tests,
trains a temporally validated LightGBM model, calibrates it, explains it
with SHAP, and turns the calibrated probabilities into a
retention-campaign expected value / breakeven analysis.""")

code("""import warnings
warnings.filterwarnings("ignore")

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.cohorts import build_cohort_retention
from src.economics import breakeven_conversion_rate, campaign_expected_profit
from src.feature_mart import build_feature_mart
from src.hypothesis_tests import chi_square_independence, correct_pvalues, two_proportion_ztest
from src.labels import derive_churn_labels
from src.modeling import FEATURE_COLUMNS, brier_before_after, calibrate_isotonic, evaluate, train_lightgbm
from src.sampling import stratified_sample
from src.shap_utils import compute_shap_values
from src.survival import fit_km_by_segment, logrank_pvalue

CUTOFFS = {"train": "2016-12-31", "val": "2017-01-31", "test": "2017-02-28"}
LABEL_HORIZON_DAYS = 30""")

md("""## 1. Data

Raw KKBox CSVs were converted to Parquet once (`data/interim/`) via
`scripts/download_data.sh` + a one-time DuckDB `COPY ... TO parquet`
pass, casting the dataset's `YYYYMMDD`-integer date columns to real
`DATE`s. `transactions.parquet` merges `transactions.csv` (history
through 2017-02-28) with `transactions_v2.csv` (extends to 2017-03-31)
-- the test-fold cutoff (2017-02-28) needs that extra month of forward
visibility to observe whether a member renews within the 30-day churn
window; `transactions.csv` alone ends exactly on the cutoff itself.""")

code("""con = duckdb.connect()
con.execute('SET memory_limit="6GB"')
con.execute("SET threads TO 6")
con.execute("SET temp_directory='data/interim/duckdb_tmp'")
con.execute("CREATE VIEW members AS SELECT * FROM read_parquet('data/interim/members.parquet')")
con.execute("CREATE VIEW transactions AS SELECT * FROM read_parquet('data/interim/transactions.parquet')")
con.execute("CREATE VIEW user_logs AS SELECT * FROM read_parquet('data/interim/user_logs.parquet')")

members_df = con.execute("SELECT * FROM members").df()
members_df["registration_init_time"] = pd.to_datetime(members_df["registration_init_time"])
print(f"members: {len(members_df):,} rows")""")

md("## 2. EDA: demographics sanity check")

code("""fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
members_df["bd"].clip(-50, 150).hist(bins=60, ax=axes[0])
axes[0].set_title("bd (age) raw -- note implausible values outside [10, 80]")
members_df["bd"].clip(10, 80).hist(bins=71, ax=axes[1])
axes[1].set_title("bd clipped to [10, 80]")
plt.tight_layout()
plt.savefig("data/processed/fig_bd_distribution.png", dpi=100)
plt.show()

missingness = members_df.isna().mean().sort_values(ascending=False)
print("missingness per column:\\n", missingness)""")

code("""official_train = con.execute("SELECT * FROM read_csv_auto('data/raw/train.csv')").df()
print(f"official train.csv (Kaggle, Jan-2017 cutoff): {len(official_train):,} rows, "
      f"churn rate {official_train['is_churn'].mean():.3%}")""")

md("""## 3. Feature mart + labels for the three temporal cutoffs

Candidate scoping: a member only has a churn/retain decision to make
around a cutoff if their subscription is actually coming up for renewal
then. Restricting to members whose `expire_at_cutoff` falls in
`[cutoff, cutoff + 28 days)` reproduces Kaggle's own `train.csv`
population almost exactly (validated separately: ~956k candidates at
~7.7% churn vs. train.csv's 993k at 6.4%, 95% label agreement on the
overlap) -- without it the population includes everyone with *any*
transaction history, most of them mid-subscription with nothing to
decide yet, which inflates apparent churn to ~50%.""")

code("""tx_all = con.execute("SELECT msno, transaction_date, membership_expire_date FROM transactions").df()
tx_all["transaction_date"] = pd.to_datetime(tx_all["transaction_date"])
tx_all["membership_expire_date"] = pd.to_datetime(tx_all["membership_expire_date"])
print(f"transactions: {len(tx_all):,} rows, {tx_all['transaction_date'].min()} to {tx_all['transaction_date'].max()}")

folds = []
for fold_name, cutoff in CUTOFFS.items():
    mart = build_feature_mart(con, cutoff)
    labels = derive_churn_labels(tx_all, pd.Timestamp(cutoff), label_horizon_days=LABEL_HORIZON_DAYS)
    cutoff_ts = pd.Timestamp(cutoff)
    labels = labels[
        (labels["expire_at_cutoff"] >= cutoff_ts)
        & (labels["expire_at_cutoff"] < cutoff_ts + pd.Timedelta(days=28))
    ]
    merged = mart.merge(labels[["msno", "is_churn"]], on="msno", how="inner")
    merged["fold"] = fold_name
    folds.append(merged)
    print(f"{fold_name} ({cutoff}): {len(merged):,} candidates, churn rate {merged['is_churn'].mean():.3%}")

population = pd.concat(folds, ignore_index=True)
del folds
print(f"combined population: {len(population):,} rows")""")

code("""extra = con.execute('''
    SELECT msno, city, LEAST(GREATEST(bd, 10), 80) AS bd_clipped, gender, registered_via
    FROM members
''').df()
population = population.merge(extra, on="msno", how="left")
population.head()""")

md("## 4. Stratified sample to 300k rows (runtime tradeoff)")

code("""population["has_recent_activity"] = population["active_days_last_30"] > 0
sample = stratified_sample(population, "has_recent_activity", sample_size=300_000, random_state=42)
sample.to_parquet("data/processed/sampled_population.parquet", index=False)
print(f"sampled population: {len(sample):,} rows")
print(sample.groupby("fold")["is_churn"].agg(["count", "mean"]))""")

md("## 5. Cohort retention")

code("""tx_all["month"] = tx_all["transaction_date"].values.astype("datetime64[M]")
active_periods = tx_all[["msno", "month"]].drop_duplicates().rename(columns={"month": "period"})
cohort_retention = build_cohort_retention(members_df, active_periods)
cohort_retention.to_parquet("data/processed/cohort_retention.parquet")

recent_cohorts = cohort_retention.tail(18).iloc[:, :12]
fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(recent_cohorts.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_yticks(range(len(recent_cohorts.index)))
ax.set_yticklabels([str(p) for p in recent_cohorts.index])
ax.set_xticks(range(recent_cohorts.shape[1]))
ax.set_xlabel("months since signup")
ax.set_title("Cohort retention (last 18 monthly cohorts)")
plt.colorbar(im, ax=ax, label="retention rate")
plt.tight_layout()
plt.savefig("data/processed/fig_cohort_retention.png", dpi=100)
plt.show()""")

md("## 6. Survival analysis (Kaplan-Meier + log-rank), test-fold snapshot")

code("""test_fold = sample[sample["fold"] == "test"].merge(
    members_df[["msno", "registration_init_time"]], on="msno", how="left"
)
test_fold["duration_days"] = (pd.Timestamp(CUTOFFS["test"]) - test_fold["registration_init_time"]).dt.days

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
survival_pvalues = {}
for ax, segment_col in zip(axes, ["is_auto_renew", "registered_via"]):
    seg = test_fold[segment_col].astype("string")
    top_groups = seg.value_counts().head(3).index.tolist()
    fitters = fit_km_by_segment(
        test_fold["duration_days"], test_fold["is_churn"],
        seg.where(seg.isin(top_groups)),
    )
    for label, kmf in fitters.items():
        kmf.plot_survival_function(ax=ax)
    ax.set_title(f"KM survival by {segment_col}")
    ax.set_xlabel("tenure (days)")
    if len(top_groups) >= 2:
        p = logrank_pvalue(test_fold["duration_days"], test_fold["is_churn"], seg, top_groups[0], top_groups[1])
        survival_pvalues[segment_col] = p
        print(f"log-rank {segment_col} ({top_groups[0]} vs {top_groups[1]}): p={p:.4g}")
plt.tight_layout()
plt.savefig("data/processed/fig_survival_curves.png", dpi=100)
plt.show()""")

md("## 7. Hypothesis testing with Benjamini-Hochberg correction")

code("""p_values, test_labels = [], []

auto_renew_churn = test_fold.groupby("is_auto_renew")["is_churn"].agg(["sum", "count"])
if len(auto_renew_churn) == 2:
    _, p = two_proportion_ztest(tuple(auto_renew_churn["sum"]), tuple(auto_renew_churn["count"]))
    p_values.append(p); test_labels.append("auto_renew_vs_churn")

reg_via_table = pd.crosstab(test_fold["registered_via"], test_fold["is_churn"])
if reg_via_table.shape[0] > 1:
    _, p = chi_square_independence(reg_via_table)
    p_values.append(p); test_labels.append("registered_via_vs_churn")

price_tercile = pd.qcut(test_fold["plan_list_price"], 3, labels=["low", "mid", "high"], duplicates="drop")
price_table = pd.crosstab(price_tercile, test_fold["is_churn"])
if price_table.shape[0] > 1:
    _, p = chi_square_independence(price_table)
    p_values.append(p); test_labels.append("price_tercile_vs_churn")

reject_flags = correct_pvalues(p_values)
hypothesis_results = pd.DataFrame({"test": test_labels, "p_value": p_values, "significant_after_BH": reject_flags})
hypothesis_results.to_parquet("data/processed/hypothesis_test_results.parquet", index=False)
hypothesis_results""")

md("""## 8. Modeling: temporal LightGBM + isotonic calibration

Train/val/test come from three *non-overlapping monthly cutoffs*
(Dec/Jan/Feb), not a random split -- a random split would leak future
activity/renewal patterns backward into training.""")

code("""train_df = sample[sample["fold"] == "train"]
val_df = sample[sample["fold"] == "val"]
test_df = sample[sample["fold"] == "test"]
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{name}: {len(df):,} rows, churn rate {df['is_churn'].mean():.3%}")

model = train_lightgbm(train_df, val_df)
metrics = evaluate(model, test_df)
print(f"test PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")""")

code("""from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

p_churn_raw = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
RocCurveDisplay.from_predictions(test_df["is_churn"], p_churn_raw, ax=axes[0])
axes[0].set_title(f"ROC (AUC={metrics['roc_auc']:.3f})")
PrecisionRecallDisplay.from_predictions(test_df["is_churn"], p_churn_raw, ax=axes[1])
axes[1].set_title(f"Precision-Recall (AUC={metrics['pr_auc']:.3f})")
plt.tight_layout()
plt.savefig("data/processed/fig_pr_roc.png", dpi=100)
plt.show()

decile = pd.DataFrame({"p_churn": p_churn_raw, "is_churn": test_df["is_churn"].values})
decile["decile"] = pd.qcut(decile["p_churn"], 10, labels=False, duplicates="drop")
lift_table = decile.groupby("decile")["is_churn"].agg(["sum", "count"])
lift_table["pct_of_total_churn"] = lift_table["sum"] / lift_table["sum"].sum()
lift_table = lift_table.sort_index(ascending=False)
lift_table["cum_pct_of_total_churn"] = lift_table["pct_of_total_churn"].cumsum()
lift_table""")

md("## 9. Calibration")

code("""calibrated = calibrate_isotonic(model, val_df)
brier = brier_before_after(model, calibrated, test_df)
print(f"Brier before={brier['brier_before']:.4f}  after={brier['brier_after']:.4f}")

p_churn_calibrated = calibrated.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(5, 5))
for label, proba in [("raw", p_churn_raw), ("isotonic-calibrated", p_churn_calibrated)]:
    frac_pos, mean_pred = calibration_curve(test_df["is_churn"], proba, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed churn rate")
ax.legend()
plt.tight_layout()
plt.savefig("data/processed/fig_calibration.png", dpi=100)
plt.show()""")

md("## 10. SHAP interpretation")

code("""shap_values = compute_shap_values(model, test_df[FEATURE_COLUMNS])
shap.summary_plot(shap_values, test_df[FEATURE_COLUMNS], show=False)
plt.tight_layout()
plt.savefig("data/processed/fig_shap_summary.png", dpi=100)
plt.show()

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS).sort_values(ascending=False)
mean_abs_shap.head(5)""")

md("""## 11. Retention-campaign economics

Sweeping the contact-volume threshold (top-k% by calibrated churn risk)
against `src/economics.py`'s EV formula finds the profit-maximizing
contact volume and its breakeven conversion rate -- the number this
project's thesis says the business should actually be tracking.""")

code("""ARPU, AVG_LIFETIME_MONTHS, CONTACT_COST, ASSUMED_CONVERSION = 4.99, 21, 3.0, 0.15

p_series = pd.Series(p_churn_calibrated, index=test_df.index)
ks = np.arange(0.01, 1.01, 0.01)
profits = [
    campaign_expected_profit(p_series, k, ASSUMED_CONVERSION, ARPU, AVG_LIFETIME_MONTHS, CONTACT_COST)
    for k in ks
]
best_idx = int(np.argmax(profits))
best_k, best_profit = ks[best_idx], profits[best_idx]
breakeven = breakeven_conversion_rate(p_series, best_k, ARPU, AVG_LIFETIME_MONTHS, CONTACT_COST)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ks * 100, profits)
ax.axvline(best_k * 100, color="green", linestyle="--", label=f"best top-{best_k:.0%}")
ax.set_xlabel("contact top K% of scored base")
ax.set_ylabel(f"expected profit ($, at {ASSUMED_CONVERSION:.0%} assumed conversion)")
ax.set_title("Campaign expected profit vs. contact volume")
ax.legend()
plt.tight_layout()
plt.savefig("data/processed/fig_campaign_economics.png", dpi=100)
plt.show()

print(f"best contact volume: top {best_k:.0%} of scored base")
print(f"expected profit at that volume (assuming {ASSUMED_CONVERSION:.0%} conversion): ${best_profit:,.0f}")
print(f"breakeven conversion rate at that volume: {breakeven:.1%}")""")

md("## 12. Write scored feature mart (feeds the Streamlit dashboard)")

code("""scored = test_df.copy()
scored["p_churn_raw"] = p_churn_raw
scored["p_churn_calibrated"] = p_churn_calibrated
scored.to_parquet("data/processed/scored_feature_mart.parquet", index=False)
print(f"wrote {len(scored):,} rows to data/processed/scored_feature_mart.parquet")""")

nb["cells"] = cells
nbf.write(nb, "churn_retention_analysis.ipynb")
print("wrote churn_retention_analysis.ipynb with", len(cells), "cells")
