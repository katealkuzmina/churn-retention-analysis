import pandas as pd

from src.labels import derive_churn_labels


def test_derive_churn_labels_marks_renewal_vs_non_renewal():
    transactions = pd.DataFrame({
        "msno": ["A", "A", "B"],
        "transaction_date": pd.to_datetime(["2016-12-01", "2016-12-20", "2016-12-01"]),
        "membership_expire_date": pd.to_datetime(["2016-12-15", "2017-01-15", "2016-12-15"]),
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
    })
    cutoff = pd.Timestamp("2016-12-15")

    labels = derive_churn_labels(transactions, cutoff, label_horizon_days=30)
    result = labels.set_index("msno")["is_churn"].to_dict()

    assert result == {"C": 0}
