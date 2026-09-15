# Subscription Churn: From Prediction to Retention Economics

Churn models usually stop at AUC. This project goes one step further: who
should actually get a retention call, and at what point does that call stop
paying for itself.

Dataset: [KKBox Churn Prediction Challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge)
(WSDM 2018 Kaggle competition) — member profiles, subscription transactions,
and daily listening activity for a real music-streaming service. Chosen over
a synthetic dataset so the eval protocol is comparable to a known public
benchmark, not invented.

## Problem statement

Given a member's subscription and activity history up to a cutoff date,
predict whether they'll churn (fail to renew within 30 days of their
subscription expiring), and turn that prediction into a concrete answer to
a business question: which members should a retention campaign contact, and
at what campaign conversion rate does contacting them stop being profitable.

## Data

This repo's code is MIT-licensed (see `LICENSE`); the KKBox dataset
itself is not — it stays under Kaggle's WSDM 2018 competition rules
regardless of this repo's license.

- `members_v3.csv` (6.77M rows) — demographics: city, age (`bd`, noisy —
  clipped to `[10, 80]` in EDA), gender (65% missing — not everyone filled
  it in), registration channel, registration date.
- `transactions.csv` + `transactions_v2.csv` (22.98M rows merged,
  2015-01-01 to 2017-03-31) — payment method, plan price, auto-renew flag,
  transaction date, membership expiry date, cancellation flag.
  `transactions.csv` alone ends exactly on 2017-02-28; the `_v2` extension
  (through 2017-03-31) is required to correctly label the Feb-28 cutoff,
  since that label needs to look 30 days into the future.
- `user_logs.csv` + `user_logs_v2.csv` (392M+ rows, ~30GB raw) — daily
  listening activity: play counts by completion bucket, unique tracks,
  total seconds played.

**Scale decisions:**
- All three raw CSVs were converted once to Parquet via DuckDB (30GB → 10GB,
  columnar, re-read three times for the three cutoffs below without
  re-parsing text each time). The dataset's `YYYYMMDD`-integer date columns
  (`registration_init_time`, `transaction_date`, `membership_expire_date`,
  `date`) are cast to real `DATE`s during that conversion.
- The full candidate population per cutoff (~820k-840k members, see below)
  is stratified-sampled down to 300k rows total across the three folds, to
  keep the notebook re-runnable in well under an hour. Sampling is
  stratified on "has any listening activity in the 30 days before cutoff"
  so the sample doesn't skew toward inactive members; seed=42 for
  reproducibility.

## Label definition and temporal cutoffs

A member is **churned** if, for a subscription active going into the
cutoff, no transaction extends `membership_expire_date` to within 30 days
of that subscription's expiry. This matches KKBox's own reference labeller
(`data/raw/WSDMChurnLabeller.scala`, shipped with the competition data) —
notably, the "renewal gap" there is *often negative*: auto-renewal charges
routinely land on or before the old expiry date, not after it. An earlier
version of `derive_churn_labels` required the renewal transaction to be
dated strictly after the old expiry and, against real data, misclassified
80-99% of members as churned. Fixed in `src/labels.py` — see its docstring
and the regression test in `tests/test_labels.py`.

**Candidate scoping:** a member only has a churn/retain decision to make
around a given cutoff if their subscription is actually coming up for
renewal then. Restricting each cutoff's population to members whose
subscription expiry falls in `[cutoff, cutoff + 28 days)` reproduces
Kaggle's own `train.csv` population closely: ~956k candidates at ~7.7%
churn (this project's exact window) vs. `train.csv`'s 993k at 6.4%, with
95% label agreement on the overlapping members. Without this filter the
population includes every member with any transaction history — most of
them mid-subscription with nothing to decide yet — which inflates apparent
churn to ~50%.

**Three monthly cutoffs**, used as non-overlapping temporal train/val/test
folds (not a random split, which would leak future renewal/activity
patterns backward into training):

Cutoffs are one month earlier than an initial draft used (which had the test fold's 30-day churn horizon running past 2017-03-31, the last date the data covers — see the regression test in `tests/test_candidates.py`).

| Fold | Cutoff | Candidates | Churn rate |
|---|---|---|---|
| Train | 2016-11-30 | 839,356 | 7.674% |
| Validation | 2016-12-31 | 820,637 | 4.082% |
| Test | 2017-01-31 | 841,374 | 3.721% |

(For reference: Kaggle's own `train.csv`, a similar Jan-2017-cutoff
population defined slightly differently, reports 992,931 members at 6.39%
churn — this project's numbers are in the same ballpark; the gap is
attributable to the exact candidate-window and lookback choices above,
called out here rather than tuned away.)

## SQL feature mart

`sql/feature_mart.sql` (DuckDB, parameterized by cutoff date) computes, per
member: tenure, current subscription state (auto-renew, payment method,
plan price, discount flag), 90-day transaction/cancellation counts via conditional aggregates (`COUNT(*)/SUM(...) FILTER
(WHERE ...)`) over `transactions`, and 30-day activity aggregates (active days, total seconds
played, and a 30-vs-prior-30-day activity trend ratio) the same way over `user_logs`. The
last-transaction-before-cutoff lookup uses a genuine window function (`ROW_NUMBER() OVER (...)`);
the rolling aggregates do not. Every subquery filters
strictly before its cutoff — no feature ever sees data on/after its own
cutoff. `src/feature_mart.py` runs it; `tests/test_feature_mart.py` pins
the exact window-function arithmetic against a hand-computed fixture.
Demographic columns (city, age, gender, registration channel) are joined in
separately in the notebook.

## Cohort retention and survival analysis

Monthly signup cohorts, denominator scoped to the renewal-candidate
population (not every KKBox registrant ever — using the raw 6.77M-row
`members` table as the denominator was an earlier bug that made month-0
retention read ~12%). Month-0 retention for cohorts registered *within* the
transactions data's coverage window (2015-01 through 2017-03) is
near-universal: mean 80.5%, range 60.1–100.0% across the 26 such cohorts
(`data/processed/cohort_retention.parquet`, column `0`).

Cohorts registered before 2015-01 — about 56% of the candidate population by
registration date — show gaps (`NaN`), not zero retention, in their early
`months_since_signup` values: the transactions data simply doesn't extend
back that far for them (left-censoring/left-truncation), so those months were
never observed at all, let alone observed-and-empty. Symmetrically, a
cohort's latest months are `NaN` once the data hasn't caught up to them yet
(e.g. the 2017-03 cohort only has a month-0 value). `build_cohort_retention`
(`src/cohorts.py`) takes both a `min_observable_month` and
`max_observable_month` bound per cohort and only zero-fills cells inside that
window — cells outside it, on either edge, stay `NaN` rather than being
misread as "nobody retained." Beyond month 0, cohorts show the expected
shape where data exists: steep drop-off in the first few months, then a
long, slowly-decaying tail for members who stick around
(`data/processed/fig_cohort_retention.png`).

Kaplan-Meier survival curves (`lifelines`), segmented by auto-renew status
and registration channel. Duration is time from registration to the renewal
decision at the test cutoff (`expire_at_cutoff - registration_init_time`);
event is whether that renewal decision ended in churn (`is_churn`) — this
ties duration to the same renewal decision the event describes, rather than
an unrelated calendar-tenure snapshot as of a fixed date. Segments show
materially different survival profiles, confirmed with log-rank tests
(auto-renew: p≈0; registration channel: p=8.124e-19).
See `data/processed/fig_survival_curves.png`.

## Hypothesis testing

Three tests on the test-fold population (auto-renew vs. churn: two-proportion
z-test; registration channel vs. churn and plan-price-tercile vs. churn:
chi-square independence), with Benjamini-Hochberg FDR correction applied
across the family (`src/hypothesis_tests.py`, `statsmodels`). All three
remain significant after correction — at a 300k-row sample size, p-values
underflow to 0.0 for effects this large.

## Model and temporal validation

LightGBM (`scale_pos_weight` set to the train-fold imbalance ratio, not
resampling), early-stopped on the validation fold, evaluated once on the
untouched test fold:

- **PR-AUC: 0.3592** (primary metric — the positive class is ~4% of the
  population, so PR-AUC is the honest number; ROC-AUC alone overstates
  performance at this base rate)
- **ROC-AUC: 0.8669** (reported as the more familiar secondary number)

**Baseline comparison:** a plain logistic regression on the same features
scores PR-AUC=0.2785 — the LightGBM model's lift over that baseline is
0.0807 points, not just its absolute PR-AUC (ROC-AUC for the baseline:
0.8404).

See `data/processed/fig_pr_roc.png`.

## Calibration

Raw LightGBM probabilities feed directly into the dollar formula below, so
they need to be genuinely calibrated, not just rank-ordered. Isotonic
regression (via `sklearn`'s `FrozenEstimator`, fit on the validation fold)
cuts the test-fold **Brier score from 0.0999 to 0.0283**. Reliability
diagram: `data/processed/fig_calibration.png`.

## SHAP interpretation

`data/processed/fig_shap_summary.png`. Top features by mean |SHAP|:
`num_transactions_last_90d`, `is_auto_renew`, `tenure_days`,
`num_cancels_lifetime`, `payment_method_id` — auto-renew status and recent
transaction/engagement behavior dominate. Note: demographic
columns (city, age, gender, registration channel) are joined into the analysis population but are
NOT among the model's `FEATURE_COLUMNS` (see `src/modeling.py`) -- they were never given to the
model in the first place, so "demographics barely register" would be the wrong conclusion to draw
from this SHAP plot. Whether demographics matter at all is an open question this project doesn't
actually answer.

## Economics: expected value and breakeven conversion rate

`src/economics.py` turns a calibrated churn probability into a dollar
decision: `ev_per_contact = p_churn * conversion_rate * (arpu *
avg_lifetime_months * margin_rate) - contact_cost`, summed over whichever
top-k% of the scored base gets contacted. `ltv`/`ev_per_contact`/
`campaign_expected_profit`/`breakeven_conversion_rate` all take a
`margin_rate` parameter (default `1.0`, i.e. raw ARPU/revenue, for
backward compatibility) so the campaign's spend (`contact_cost`) can be
weighed against the *margin* it actually keeps, not gross revenue.

**A mentor review flagged two problems with the original headline
number:** (1) it assumed a 21-month average lifetime, which is the
base-rate across the whole member base, not the much shorter actual
lifetime of the ~33%-per-cycle-churning target group the campaign
contacts; and (2) it used raw revenue instead of margin, and was computed
on the ~12%-sampled test fold (~100k rows) without being scaled up to the
true test-fold population (~841k rows).

Both are now fixed. `scale_to_full_population(sample_profit, sample_size,
full_population_size)` rescales a profit figure computed on the sampled
fold up to what contacting the same top-k% of the *full* un-sampled
test-fold population (`full_test_population_size`, captured right after
`population` is built and before `stratified_sample` runs) would yield.
Sweeping contact volume against the test fold's calibrated probabilities
(ARPU=$4.99, contact cost=$3, 15% assumed campaign conversion rate):

- **Best contact volume: top 5%** of the scored base
- **Expected profit at that volume (21mo lifetime, 100% margin_rate,
  sampled fold): ~$17,683 — at full-test-population scale: ~$147,662**
- **Breakeven conversion rate at that volume: 6.9%** — below this,
  contacting that many people loses money regardless of how good the churn
  model's ranking is.

### Scenario table (lifetime and margin sensitivity)

Holding the same top-5% contact volume fixed, varying the lifetime
assumption and margin_rate (real output from `scripts/build_pipeline.py`,
`sample_profit` on the ~100k-row test fold, `full_population_profit`
scaled via `scale_to_full_population` to the ~841k-row full test-fold
population):

| scenario | sample_profit | full_population_profit |
|---|---:|---:|
| 21mo revenue (original assumption) | $17,682.56 | $147,661.62 |
| 12mo revenue (target-group-adjusted lifetime) | $3,626.89 | $30,287.05 |
| 12mo margin at 40% margin_rate | -$7,617.64 | -$63,612.60 |
| 6mo revenue | -$5,743.55 | -$47,962.65 |

Shortening the assumed lifetime from 21 to 12 months (closer to what the
~33%-per-cycle-churning target group actually experiences) cuts expected
profit by roughly 80%; applying a realistic 40% margin_rate on top of
that 12-month lifetime flips the campaign from profitable to a loss; and
at a 6-month lifetime the campaign is unprofitable even at 100%
margin_rate. **The headline dollar figure above is reported at
full-test-population scale (via `scale_to_full_population`) and, where
`margin_rate` is applied, on margin rather than raw ARPU — the original
$13.8k/$2.3k figures from an earlier, buggy pipeline run no longer
apply and should not be cited.**

**Uplift caveat:** this EV model assumes the campaign's `conversion_rate`
applies uniformly to everyone contacted. In practice, a retention offer's
effect (uplift) varies by member — some would've stayed anyway, some
can't be saved regardless of the offer. Properly targeting contact volume
would use an uplift model (e.g. two-model or a causal-tree approach
trained on historical campaign A/B data) to target *persuadable* members,
not just *highest-churn-risk* ones. Not implemented here — flagged as the
natural next step, since it changes who gets contacted, not just the
breakeven math.

See `data/processed/fig_campaign_economics.png`.

## Dashboard

`dashboard/app.py` (Streamlit) — four pages: Overview, Survival &
hypothesis tests, Model performance, and a live **Campaign economics**
page where ARPU/lifetime/contact-cost/conversion-rate/margin-rate are sliders and the
expected-profit and breakeven-conversion-rate numbers recompute instantly
via the same `src/economics.py` functions used in the notebook (no
duplicated logic between the two).

```bash
uv run streamlit run dashboard/app.py
```

## How to reproduce

```bash
uv sync
./scripts/download_data.sh          # needs `kaggle` CLI credentials + accepted competition rules
uv run python scripts/convert_to_parquet.py   # one-time raw CSV -> Parquet conversion with date casting
uv run python scripts/make_notebook.py
PYTHONPATH=. uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=540 churn_retention_analysis.ipynb \
  --output churn_retention_analysis.ipynb
uv run streamlit run dashboard/app.py
```

`scripts/build_pipeline.py` is a non-interactive equivalent of the notebook
(same `src/` calls, no plots) — useful for quick end-to-end reruns while
iterating; the notebook is the primary, documented artifact.

Run the test suite (fast — no data download needed for most of it):

```bash
uv run pytest
```

One test (`tests/test_data_smoke.py`) is skipped on a truly fresh clone, before
`scripts/download_data.sh` has put the raw CSVs in place -- it passes once they're there.
`data/processed/scored_feature_mart.parquet` is committed to the repo, so the dashboard-smoke
tests run (and pass) even on a fresh clone without a full pipeline re-run.

## Limitations

- The 300k-row sample (from a ~2.5M-row combined candidate population) is a
  runtime tradeoff, disclosed above, not a modeling choice — a full run
  would use the entire population.
- The campaign `conversion_rate` (15% in the notebook, adjustable in the
  dashboard) is an **assumed planning input, not a measured one** — pinning
  it down for real would need an actual A/B test of the retention campaign.
  The breakeven-conversion-rate number sidesteps needing to know it exactly:
  it tells you the threshold the real, measured conversion rate needs to
  clear.
- This project's churn rate (~4%) runs somewhat below Kaggle's own
  `train.csv` (6.4%) for a comparable population — attributable to this
  project's exact candidate-window and lookback choices (documented above),
  not tuned to match.
