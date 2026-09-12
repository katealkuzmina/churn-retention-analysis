from pathlib import Path

import duckdb
import pytest

RAW = Path("data/raw")
REQUIRED = ["members_v3.csv", "transactions.csv", "user_logs.csv", "train.csv"]

pytestmark = pytest.mark.skipif(
    not RAW.exists() or not all((RAW / f).exists() for f in REQUIRED),
    reason="raw KKBox data not downloaded locally",
)


def test_raw_tables_have_expected_columns():
    con = duckdb.connect()
    members_cols = set(
        con.sql(f"SELECT * FROM read_csv_auto('{RAW}/members_v3.csv') LIMIT 0").columns
    )
    assert {"msno", "city", "bd", "gender", "registered_via", "registration_init_time"} <= members_cols

    tx_cols = set(
        con.sql(f"SELECT * FROM read_csv_auto('{RAW}/transactions.csv') LIMIT 0").columns
    )
    assert {
        "msno", "payment_method_id", "is_auto_renew",
        "transaction_date", "membership_expire_date", "is_cancel",
    } <= tx_cols
