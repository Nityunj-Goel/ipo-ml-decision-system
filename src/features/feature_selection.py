import pandas as pd
from configs.feature_config import FINAL_FEATURES, TARGET


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FINAL_FEATURES]