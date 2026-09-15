import sys
from pathlib import Path

# Make `src` importable regardless of the cwd streamlit was launched from
# (e.g. `streamlit run dashboard/app.py` from the project root vs. from
# inside dashboard/) -- PYTHONPATH is easy to forget to set by hand.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import pandas as pd
import streamlit as st

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
    st.caption(
        "Cohort retention heatmap and KPI tiles render from "
        "data/processed/cohort_retention.parquet (Task 13)."
    )

elif page == "Survival & hypothesis tests":
    st.title("Survival & hypothesis tests")
    st.caption(
        "KM curves by segment and the corrected hypothesis-test table render "
        "from data/processed/survival_results.parquet (Task 13)."
    )

elif page == "Model performance":
    st.title("Model performance")
    st.caption(
        "PR/ROC curves, calibration diagram, decile lift table, and SHAP "
        "summary render from data/processed/model_eval.parquet (Task 13)."
    )

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
