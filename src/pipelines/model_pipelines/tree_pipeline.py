"""Tree-based model pipeline: minimal preprocessing → RandomForestClassifier."""
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from configs.feature_config import RAW_FEATURES, FINAL_FEATURES

# Skewed columns — log-transform helps even for trees (reduces outlier influence on splits)
_LOG_COLS = [
    RAW_FEATURES["issue_amount"],
]

# Everything else passes through untouched (trees don't need scaling)
_PASSTHROUGH_COLS = [f for f in FINAL_FEATURES if f not in _LOG_COLS]


def get_tree_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p, validate=False), _LOG_COLS),
        ("pass", "passthrough", _PASSTHROUGH_COLS),
    ], remainder="drop")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
