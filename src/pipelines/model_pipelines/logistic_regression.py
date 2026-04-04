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
]

# Everything in FINAL_FEATURES that isn't log-transformed or boolean
_NUMERIC_COLS = [f for f in FINAL_FEATURES if f not in _LOG_COLS + [DERIVED_FEATURES["is_gmp_missing"], RAW_FEATURES['gmp']]]

def _signed_log_transform(x):
    return np.sign(x) * np.log1p(np.abs(x))


def get_logistic_regression_pipeline(**log_reg_kwargs) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("signed_log", Pipeline([
            ("signed_log_transform", FunctionTransformer(_signed_log_transform, validate=False)),
            ("scale", StandardScaler()),
        ]), [RAW_FEATURES['gmp']]),
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