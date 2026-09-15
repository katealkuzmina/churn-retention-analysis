"""One-time raw CSV -> Parquet conversion for the three KKBox source
tables, casting YYYYMMDD-integer date columns to real DATEs. Run once
after scripts/download_data.sh; scripts/build_pipeline.py and the
notebook both read from data/interim/*.parquet, not the raw CSVs.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

RAW = "data/raw"
INTERIM = "data/interim"


def main() -> None:
    # data/interim/ is gitignored (only .gitkeep is tracked) -- create it
    # defensively in case it's missing, since DuckDB's COPY TO does not
    # create parent directories itself.
    Path(INTERIM).mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute('SET memory_limit="4GB"')

    con.execute(f"""
        COPY (
            SELECT
                msno, city, bd, gender, registered_via,
                strptime(CAST(registration_init_time AS VARCHAR), '%Y%m%d')::DATE
                    AS registration_init_time
            FROM read_csv_auto('{RAW}/members_v3.csv')
        ) TO '{INTERIM}/members.parquet' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT
                msno, payment_method_id, payment_plan_days, plan_list_price,
                actual_amount_paid, is_auto_renew,
                strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS transaction_date,
                strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS membership_expire_date,
                is_cancel
            FROM read_csv_auto('{RAW}/transactions.csv')
            UNION ALL
            SELECT
                msno, payment_method_id, payment_plan_days, plan_list_price,
                actual_amount_paid, is_auto_renew,
                strptime(CAST(transaction_date AS VARCHAR), '%Y%m%d')::DATE AS transaction_date,
                strptime(CAST(membership_expire_date AS VARCHAR), '%Y%m%d')::DATE AS membership_expire_date,
                is_cancel
            FROM read_csv_auto('{RAW}/transactions_v2.csv')
        ) TO '{INTERIM}/transactions.parquet' (FORMAT PARQUET)
    """)

    con.execute(f"""
        COPY (
            SELECT
                msno,
                strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS date,
                num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
            FROM read_csv_auto('{RAW}/user_logs.csv')
            UNION ALL
            SELECT
                msno,
                strptime(CAST(date AS VARCHAR), '%Y%m%d')::DATE AS date,
                num_25, num_50, num_75, num_985, num_100, num_unq, total_secs
            FROM read_csv_auto('{RAW}/user_logs_v2.csv')
        ) TO '{INTERIM}/user_logs.parquet' (FORMAT PARQUET)
    """)

    print("wrote data/interim/{members,transactions,user_logs}.parquet")


if __name__ == "__main__":
    main()
