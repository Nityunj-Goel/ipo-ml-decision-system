# Needs manual review and understanding
# Study sklearn pipelines

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

# Use RAW_FEATURES to get colNames
NUMERIC_LOG_COLS = [
    "Assets (Rs.cr.)",
    "Revenue (Rs.cr.)",
    "Issue Amount (Rs.cr.)",
]

NUMERIC_COLS = [
    "P/E (x) Pre-IPO",
    "Price to Book Value",
    "PAT Margin %",
    "ROCE",
    "ROE"
]

RATIO_COLS = [
    "qib_ratio",
    "retail_ratio",
    "nii_ratio"
]

BINARY_COLS = [
    "is_gmp_missing"
]

def get_logistic_pipeline():

    log_transformer = Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    ratio_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    binary_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("log", log_transformer, NUMERIC_LOG_COLS),
            ("num", numeric_transformer, NUMERIC_COLS),
            ("ratio", ratio_transformer, RATIO_COLS),
            ("bin", binary_transformer, BINARY_COLS),
        ],
        remainder="drop"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    return pipeline