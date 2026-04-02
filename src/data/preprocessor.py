import pandas as pd
from typing import Literal, Sequence
from configs.feature_config import RAW_FEATURES

redundant_cols = [
    RAW_FEATURES['issue_open_date'],
    RAW_FEATURES['closing_date'],
    RAW_FEATURES['id'],
    RAW_FEATURES['company'],
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

def preprocess(df: pd.DataFrame, mode: Literal['strict', 'unstrict'] = 'strict') -> pd.DataFrame:
    df = df.copy()
    _clean(df, mode)
    _handle_nans(df)

    return df

def _clean(df: pd.DataFrame, mode: Literal['strict', 'unstrict']) -> pd.DataFrame:
    # Handle cases when data quality issues are different from training set in inference mode (e.x. logging, raise/coerce exceptions)
    df = df.drop(columns=redundant_cols, errors="ignore")
    _clean_dates(df, mode)
    #Add Clean percentage columns step if being used
    return df

def _clean_dates(df: pd.DataFrame, mode: Literal['strict', 'unstrict']):
    d_cols = [c for c in date_cols if c in df.columns]

    for c in d_cols:
        df[c] = pd.to_datetime(
            df[c],
            errors="raise" if mode == 'strict' else "coerce"
        ).dt.normalize()

def _clean_percent(df: pd.DataFrame, mode: Literal['strict', 'unstrict']):
    p_cols = [c for c in percent_columns if c in df.columns]
    df[p_cols] = df[p_cols].apply(
        lambda s: pd.to_numeric(
            s.astype("string").str.replace("%", "", regex=False),
            errors="raise" if mode == 'strict' else "coerce"
        )
    )

def _handle_nans(df: pd.DataFrame):
    nan_cols = [
        RAW_FEATURES['qib'],
        RAW_FEATURES['price_band_high'],
        RAW_FEATURES['price_band_low'],
        RAW_FEATURES["issue_amount"]
    ]
    for c in nan_cols:
        df[c] = df[c].fillna(df[c].median())
