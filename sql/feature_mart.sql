-- One row per msno with a membership on file as of the cutoff date.
-- Every CTE filters strictly before the cutoff -- no feature may see
-- data on/after its own cutoff (spec Sec 3). Placeholders are positional,
-- each bound to the same cutoff_date value by src/feature_mart.py.

WITH last_transaction AS (
    -- Tie-broken by every remaining column (not just transaction_date) so
    -- the "most recent transaction" pick is fully deterministic even when
    -- a member has multiple same-day transactions -- DuckDB's parallel
    -- scan does not guarantee row order, so ROW_NUMBER() on
    -- transaction_date alone silently picked a different row across runs.
    SELECT * FROM (
        SELECT t.*,
            ROW_NUMBER() OVER (
                PARTITION BY msno
                ORDER BY transaction_date DESC, membership_expire_date DESC,
                    payment_method_id, payment_plan_days, plan_list_price,
                    actual_amount_paid, is_auto_renew, is_cancel
            ) AS rn
        FROM transactions t
        WHERE t.transaction_date < CAST(? AS DATE)
    )
    WHERE rn = 1
),
transactions_before_cutoff AS (
    SELECT * FROM transactions WHERE transaction_date < CAST(? AS DATE)
),
tx_rolling AS (
    SELECT
        msno,
        COUNT(*) FILTER (WHERE transaction_date >= CAST(? AS DATE) - INTERVAL 90 DAY)
            AS num_transactions_last_90d,
        SUM(is_cancel) AS num_cancels_lifetime
    FROM transactions_before_cutoff
    GROUP BY msno
),
logs_before_cutoff AS (
    SELECT * FROM user_logs WHERE date < CAST(? AS DATE)
),
logs_rolling AS (
    SELECT
        msno,
        COUNT(*) FILTER (WHERE date >= CAST(? AS DATE) - INTERVAL 30 DAY)
            AS active_days_last_30,
        SUM(total_secs) FILTER (WHERE date >= CAST(? AS DATE) - INTERVAL 30 DAY)
            AS total_secs_last_30,
        SUM(total_secs) FILTER (
            WHERE date >= CAST(? AS DATE) - INTERVAL 60 DAY
              AND date <  CAST(? AS DATE) - INTERVAL 30 DAY
        ) AS total_secs_prior_30,
        MAX(date) AS last_log_date
    FROM logs_before_cutoff
    GROUP BY msno
)
SELECT
    m.msno,
    date_diff('day', m.registration_init_time, CAST(? AS DATE)) AS tenure_days,
    lt.is_auto_renew,
    lt.payment_method_id,
    lt.plan_list_price,
    lt.actual_amount_paid,
    (lt.actual_amount_paid < lt.plan_list_price) AS discount_flag,
    COALESCE(tr.num_transactions_last_90d, 0) AS num_transactions_last_90d,
    COALESCE(tr.num_cancels_lifetime, 0) AS num_cancels_lifetime,
    COALESCE(lr.active_days_last_30, 0) AS active_days_last_30,
    COALESCE(lr.total_secs_last_30, 0.0) AS total_secs_last_30,
    COALESCE(lr.total_secs_prior_30, 0.0) AS total_secs_prior_30,
    CASE WHEN COALESCE(lr.total_secs_prior_30, 0) = 0 THEN NULL
         ELSE lr.total_secs_last_30 / lr.total_secs_prior_30 END AS activity_trend_30d,
    date_diff('day', lr.last_log_date, CAST(? AS DATE)) AS days_since_last_log
FROM members m
JOIN last_transaction lt ON lt.msno = m.msno
LEFT JOIN tx_rolling tr ON tr.msno = m.msno
LEFT JOIN logs_rolling lr ON lr.msno = m.msno
ORDER BY m.msno;
