import os

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/processed/scored_feature_mart.parquet"),
    reason="scored feature mart not built yet (Task 13)",
)


def test_dashboard_loads_without_exception():
    at = AppTest.from_file("../dashboard/app.py")
    at.run()
    assert not at.exception


def test_campaign_economics_page_renders_metrics():
    at = AppTest.from_file("../dashboard/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Campaign economics").run()
    assert not at.exception
    assert len(at.metric) == 2


def test_overview_page_renders_cohort_heatmap_and_kpis():
    at = AppTest.from_file("../dashboard/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Overview").run()
    assert not at.exception
    assert len(at.metric) >= 2  # base churn rate + at least one more KPI


def test_survival_page_renders_km_curves_and_test_table():
    at = AppTest.from_file("../dashboard/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Survival & hypothesis tests").run()
    assert not at.exception
    # The brief's spec used `at.pyplot`, but the KM curve is rendered as a
    # pre-saved static image (st.image against the notebook-generated PNG,
    # not a live re-plot via st.pyplot) and the installed streamlit version
    # (1.63.0) has no `AppTest.pyplot` accessor at all. Check the image
    # element instead -- same intent (KM curve figure renders).
    assert len(at.image) >= 1  # KM curve figure
    assert len(at.dataframe) >= 1  # hypothesis test table


def test_model_performance_page_renders_metrics_and_shap():
    at = AppTest.from_file("../dashboard/app.py")
    at.run()
    at.sidebar.radio[0].set_value("Model performance").run()
    assert not at.exception
    assert len(at.metric) >= 2  # PR-AUC + ROC-AUC at minimum
