import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.data_cleaner import DataCleaner
from src.features.feature_engineering import build_derived_features
from src.features.feature_selection import select_features
from configs.feature_config import TARGET, CleaningMode


def get_data_pipeline(mode: CleaningMode = "strict") -> Pipeline:
    return Pipeline([
        ("cleaner", DataCleaner(mode=mode)),
        ("feature_engineer", FunctionTransformer(build_derived_features)),
        ("feature_selector", FunctionTransformer(select_features)),
    ])


def compute_target(df: pd.DataFrame, listing_gain_threshold: float = 0) -> pd.Series:
    gain_pct = (
        (df[TARGET["listing_price"]] - df[TARGET["issue_price"]])
        / df[TARGET["issue_price"]]
        * 100
    )
    return (gain_pct > listing_gain_threshold).astype(int)