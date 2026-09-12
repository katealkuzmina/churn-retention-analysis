from src.modeling import FEATURE_COLUMNS, train_lightgbm
from src.shap_utils import compute_shap_values
from tests.test_modeling import _synthetic_split


def test_compute_shap_values_matches_input_shape():
    train_df, val_df, test_df = _synthetic_split()
    model = train_lightgbm(train_df, val_df)
    shap_values = compute_shap_values(model, test_df[FEATURE_COLUMNS])
    assert shap_values.shape == (len(test_df), len(FEATURE_COLUMNS))
