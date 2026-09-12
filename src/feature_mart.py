from pathlib import Path

import duckdb
import pandas as pd

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "feature_mart.sql"


def build_feature_mart(con: duckdb.DuckDBPyConnection, cutoff_date: str) -> pd.DataFrame:
    """Run the parameterized feature mart query for one cutoff.

    `con` must already have `members`, `transactions`, and `user_logs`
    tables or views registered (DuckDB tables, or `read_csv_auto` views
    over the raw files).
    """
    sql = SQL_PATH.read_text()
    params = [cutoff_date] * sql.count("?")
    return con.execute(sql, params).df()
