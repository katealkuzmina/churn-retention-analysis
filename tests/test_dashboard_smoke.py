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
