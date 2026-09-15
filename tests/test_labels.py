import pandas as pd

from src.labels import derive_churn_labels


def test_derive_churn_labels_marks_renewal_vs_non_renewal():
    transactions = pd.DataFrame({
        "msno": ["A", "A", "B"],
        "transaction_date": pd.to_datetime(["2016-12-01", "2016-12-20", "2016-12-01"]),
        "membership_expire_date": pd.to_datetime(["2016-12-15", "2017-01-15", "2016-12-15"]),
        "is_cancel": [0, 0, 0],
    })
    cutoff = pd.Timestamp("2016-12-16")

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=30)
    result = labels.set_index("msno")["is_churn"].to_dict()

    assert result == {"A": 0, "B": 1}


def test_derive_churn_labels_treats_early_auto_renewal_as_not_churned():
    # expire_at_cutoff (Dec 31) comes from the transaction before the
    # cutoff (Dec 15). C's renewal transaction (Dec 28) lands *before*
    # that Dec 31 expiry, and after the cutoff -- the normal auto-renew
    # pattern (gap = Dec28 - Dec31 = -3 days). An earlier version of
    # derive_churn_labels required transaction_date > expire_at_cutoff
    # and misread this as churn.
    transactions = pd.DataFrame({
        "msno": ["C", "C"],
        "transaction_date": pd.to_datetime(["2016-12-01", "2016-12-28"]),
        "membership_expire_date": pd.to_datetime(["2016-12-31", "2017-01-31"]),
        "is_cancel": [0, 0],
    })
    cutoff = pd.Timestamp("2016-12-15")

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=30)
    result = labels.set_index("msno")["is_churn"].to_dict()

    assert result == {"C": 0}


def test_derive_churn_labels_ignores_cancellation_transactions():
    # D's "renewal" transaction extends membership_expire_date past
    # expire_at_cutoff, but is_cancel=1 -- a cancellation, not a renewal.
    # The old code had no is_cancel check and would flip this to
    # is_churn=0 ("didn't churn") just because the expiry date moved.
    transactions = pd.DataFrame({
        "msno": ["D", "D"],
        "transaction_date": pd.to_datetime(["2016-12-01", "2016-12-20"]),
        "membership_expire_date": pd.to_datetime(["2016-12-31", "2017-01-15"]),
        "is_cancel": [0, 1],
    })
    cutoff = pd.Timestamp("2016-12-16")

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=30)
    result = labels.set_index("msno")["is_churn"].to_dict()

    assert result == {"D": 1}


def test_derive_churn_labels_ignores_transactions_before_cutoff():
    # E's Nov-01 transaction (dated BEFORE the Dec-16 cutoff) happens to
    # have a later membership_expire_date than the actual last-before-
    # cutoff transaction (Dec-01, expiring Dec-31) -- a data ordering
    # quirk, not a real renewal. The old code had no lower bound on
    # transaction_date and would count this as a renewal.
    transactions = pd.DataFrame({
        "msno": ["E", "E"],
        "transaction_date": pd.to_datetime(["2016-11-01", "2016-12-01"]),
        "membership_expire_date": pd.to_datetime(["2017-02-01", "2016-12-31"]),
        "is_cancel": [0, 0],
    })
    cutoff = pd.Timestamp("2016-12-16")

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=30)
    result = labels.set_index("msno")["is_churn"].to_dict()

    assert result == {"E": 1}
