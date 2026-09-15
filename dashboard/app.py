import sys
from pathlib import Path

# Make `src` importable regardless of the cwd streamlit was launched from
# (e.g. `streamlit run dashboard/app.py` from the project root vs. from
# inside dashboard/) -- PYTHONPATH is easy to forget to set by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, roc_auc_score

from src.economics import breakeven_conversion_rate, campaign_expected_profit

st.set_page_config(page_title="Churn Retention Economics", layout="wide")


@st.cache_data
def load_scored_mart() -> pd.DataFrame:
    con = duckdb.connect()
    return con.sql(
        "SELECT * FROM read_parquet('data/processed/scored_feature_mart.parquet')"
    ).df()


page = st.sidebar.radio(
    "Section",
    ["Overview", "Survival & hypothesis tests", "Model performance", "Campaign economics"],
)

mart = load_scored_mart()

if page == "Overview":
    st.title("Overview")
    st.metric("Base churn rate", f"{mart['is_churn'].mean():.1%}")
    st.metric("Scored population", f"{len(mart):,}")
    st.image("data/processed/fig_cohort_retention.png", caption="Monthly cohort retention")

elif page == "Survival & hypothesis tests":
    st.title("Survival & hypothesis tests")
    st.image(
        "data/processed/fig_survival_curves.png",
        caption="Kaplan-Meier survival curves by segment",
    )
    hyp = pd.read_parquet("data/processed/hypothesis_test_results.parquet")
    st.dataframe(hyp)

elif page == "Model performance":
    st.title("Model performance")
    st.metric(
        "Test PR-AUC",
        f"{average_precision_score(mart['is_churn'], mart['p_churn_calibrated']):.3f}",
    )
    st.metric(
        "Test ROC-AUC",
        f"{roc_auc_score(mart['is_churn'], mart['p_churn_calibrated']):.3f}",
    )
    st.image("data/processed/fig_pr_roc.png", caption="PR / ROC curves")
    st.image("data/processed/fig_calibration.png", caption="Calibration reliability diagram")
    st.image("data/processed/fig_shap_summary.png", caption="SHAP summary")

elif page == "Campaign economics":
    st.title("Campaign economics")
    arpu = st.slider("ARPU ($/month)", 1.0, 20.0, 4.99)
    avg_lifetime_months = st.slider("Average lifetime (months)", 1, 60, 21)
    contact_cost = st.slider("Contact cost ($)", 0.5, 20.0, 3.0)
    conversion_rate = st.slider("Assumed campaign conversion rate", 0.0, 1.0, 0.15)
    top_k = st.slider("Contact top K% of scored base", 1, 100, 12) / 100
    margin_rate = st.slider("Margin rate (share of ARPU kept as margin)", 0.1, 1.0, 0.4)

    profit = campaign_expected_profit(
        mart["p_churn_calibrated"], top_k, conversion_rate,
        arpu, avg_lifetime_months, contact_cost, margin_rate=margin_rate,
    )
    breakeven = breakeven_conversion_rate(
        mart["p_churn_calibrated"], top_k, arpu, avg_lifetime_months, contact_cost, margin_rate=margin_rate,
    )
    st.metric("Expected campaign profit", f"${profit:,.0f}")
    st.metric("Breakeven conversion rate", f"{breakeven:.1%}")
