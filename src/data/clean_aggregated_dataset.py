import pandas as pd
from typing import Literal
from src.features.feature_config import RAW_FEATURES

redundant_cols = [
    RAW_FEATURES['issue_open_date'],
    RAW_FEATURES['closing_date']
]
percent_columns = [
    RAW_FEATURES['roce'],
    RAW_FEATURES['roe'],
    RAW_FEATURES['ronw']
]
date_cols = [
    RAW_FEATURES['ipo_start_date'],
    RAW_FEATURES['ipo_end_date'],
    RAW_FEATURES['listing_date']
]

def clean(df: pd.DataFrame, mode: Literal['inference', 'training'] = 'training') -> pd.DataFrame:
    # Handle cases when data quality issues are different from training set in inference mode (e.x. logging, raise/coerce exceptions)
    df = df.copy()
    df = df.drop(columns=redundant_cols, errors="ignore")
    clean_dates(df, mode);
    # Clean percentage columns

    return df

def clean_dates(df: pd.DataFrame, mode: Literal['inference', 'training']):
    d_cols = [c for c in date_cols if c in df.columns]

    for c in d_cols:
        df[c] = pd.to_datetime(
            df[c],
            errors="coerce" if mode == 'inference' else "raise"
        ).dt.normalize()

def clean_percent(df: pd.DataFrame, mode: Literal['inference', 'training']):
    p_cols = [c for c in percent_columns if c in df.columns]
    df[p_cols] = df[p_cols].apply(
        lambda s: pd.to_numeric(
            s.astype("string").str.replace("%", "", regex=False),
            errors="coerce" if mode == 'inference' else "raise"
        )
    )
    df.rename(
        columns={c: f"{c} (%)" for c in p_cols if not c.endswith("(%)")},
        inplace=True
    )