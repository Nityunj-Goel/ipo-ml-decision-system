import pandas as pd
from configs.feature_config import FINAL_FEATURES, TARGET


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FINAL_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    return df[[*FINAL_FEATURES, TARGET['listing_gain_perc']]]