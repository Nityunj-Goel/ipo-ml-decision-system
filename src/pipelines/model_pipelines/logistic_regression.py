"""Logistic Regression model pipeline: scaling + log-transform → LogisticRegression."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

from configs.feature_config import RAW_FEATURES, FINAL_FEATURES, DERIVED_FEATURES

# Skewed columns that benefit from log1p before scaling
_LOG_COLS = [
    RAW_FEATURES["issue_amount"],
    RAW_FEATURES["nii"],
    RAW_FEATURES["qib"],
    RAW_FEATURES["retail"],
    RAW_FEATURES["total"],
    RAW_FEATURES['gmp'],
]

# Everything in FINAL_FEATURES that isn't log-transformed or boolean
_NUMERIC_COLS = [f for f in FINAL_FEATURES if f not in _LOG_COLS + [DERIVED_FEATURES["is_gmp_missing"]]]


def get_logistic_regression_pipeline(**log_reg_kwargs) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("log", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
        ]), _LOG_COLS),
        ("num", StandardScaler(), _NUMERIC_COLS)
    ], remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**log_reg_kwargs)),
    ])