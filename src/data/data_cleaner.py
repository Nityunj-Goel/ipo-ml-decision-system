import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from configs.feature_config import RAW_FEATURES, CleaningMode

_REDUNDANT_COLS = [
    RAW_FEATURES['issue_open_date'],
    RAW_FEATURES['closing_date'],
    RAW_FEATURES['id'],
    RAW_FEATURES['company'],
    # Used for target computation but not needed for modeling. Dropping to avoid leakage.
    RAW_FEATURES['issue_price'],
    RAW_FEATURES['listing_price'],
]
_PERCENT_COLS = [
    RAW_FEATURES['roce'],
    RAW_FEATURES['roe'],
    RAW_FEATURES['ronw'],
]
_DATE_COLS = [
    RAW_FEATURES['ipo_start_date'],
    RAW_FEATURES['ipo_end_date'],
    RAW_FEATURES['listing_date'],
]
_MEDIAN_IMPUTE_COLS = [
    RAW_FEATURES['qib'],
    RAW_FEATURES['price_band_high'],
    RAW_FEATURES['price_band_low'],
    RAW_FEATURES['issue_amount'],
]


class DataCleaner(BaseEstimator, TransformerMixin):
    """Drop redundant columns, coerce dates, impute NaNs with median.

    Medians are learned from training data in fit() to avoid leakage.
    """

    def __init__(self, mode: CleaningMode = 'strict'):
        self.medians_: pd.Series | None = None
        self.mode = mode

    def fit(self, X: pd.DataFrame, y=None):
        present = [c for c in _MEDIAN_IMPUTE_COLS if c in X.columns]
        self.medians_ = X[present].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.drop(columns=_REDUNDANT_COLS, errors="ignore")
        self._clean_dates(df)
        self._impute_nans(df)
        return df

    def _clean_dates(self, df):
        for c in _DATE_COLS:
            if c in df.columns:
                df[c] = pd.to_datetime(
                    df[c],
                    errors="raise" if self.mode == 'strict' else "coerce",
                ).dt.normalize()

    def _clean_percent(self, df):
        p_cols = [c for c in _PERCENT_COLS if c in df.columns]
        df[p_cols] = df[p_cols].apply(
            lambda s: pd.to_numeric(
                s.astype("string").str.replace("%", "", regex=False),
                errors="raise" if self.mode == 'strict' else "coerce",
            )
        )

    def _impute_nans(self, df):
        for col, med in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)
