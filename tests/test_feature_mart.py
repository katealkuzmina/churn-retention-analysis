import duckdb
import pandas as pd

from src.feature_mart import build_feature_mart


def _fixture_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE members (msno VARCHAR, registration_init_time DATE)")
    con.execute("INSERT INTO members VALUES ('U1', '2016-01-01')")

    con.execute("""
        CREATE TABLE transactions (
            msno VARCHAR, payment_method_id INTEGER, payment_plan_days INTEGER,
            plan_list_price DOUBLE, actual_amount_paid DOUBLE, is_auto_renew INTEGER,
            transaction_date DATE, membership_expire_date DATE, is_cancel INTEGER
        )
    """)
    con.execute("""
        INSERT INTO transactions VALUES
        ('U1', 40, 30, 149.0, 149.0, 1, '2016-11-01', '2016-11-30', 0),
        ('U1', 40, 30, 149.0, 100.0, 1, '2016-12-01', '2016-12-31', 0)
    """)

    con.execute("CREATE TABLE user_logs (msno VARCHAR, date DATE, total_secs DOUBLE)")
    con.execute("""
        INSERT INTO user_logs VALUES
        ('U1', '2016-12-05', 3600.0),
        ('U1', '2016-12-20', 1800.0),
        ('U1', '2016-11-05', 7200.0)
    """)
    return con


def test_build_feature_mart_computes_core_columns():
    con = _fixture_connection()
    mart = build_feature_mart(con, "2017-01-01")

    assert len(mart) == 1
    row = mart.iloc[0]
    assert row["msno"] == "U1"
    assert row["tenure_days"] == 366  # 2016 is a leap year
    assert row["is_auto_renew"] == 1
    assert bool(row["discount_flag"]) is True  # last tx before cutoff: 100 < 149
    assert row["active_days_last_30"] == 2  # Dec 5 and Dec 20
    assert row["total_secs_last_30"] == 3600.0 + 1800.0
    assert row["total_secs_prior_30"] == 7200.0
    assert row["activity_trend_30d"] == (3600.0 + 1800.0) / 7200.0


def test_build_feature_mart_nulls_activity_trend_when_prior_window_is_near_empty():
    # Fresh msno (U2) isolated from U1's fixture data, with its own
    # membership row -- the CASE guard needs enough prior-30d signal
    # (>= 60s) to compute a ratio at all; below that it's NULL, same
    # as a true-zero denominator.
    con = _fixture_connection()
    con.execute("INSERT INTO members VALUES ('U2', '2016-01-01')")
    con.execute("""
        INSERT INTO transactions VALUES
        ('U2', 40, 30, 149.0, 149.0, 1, '2016-12-01', '2016-12-31', 0)
    """)
    con.execute("""
        INSERT INTO user_logs VALUES
        ('U2', '2016-12-20', 500000.0),
        ('U2', '2016-11-05', 5.0)
    """)
    mart = build_feature_mart(con, "2017-01-01")
    row = mart[mart["msno"] == "U2"].iloc[0]
    assert pd.isna(row["activity_trend_30d"])  # prior-30 total (5.0s) is below the 60s floor
