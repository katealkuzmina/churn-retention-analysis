import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """TreeExplainer SHAP values for a fitted LightGBM binary classifier,
    for the positive (churn) class."""
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)
    return raw[1] if isinstance(raw, list) else raw
