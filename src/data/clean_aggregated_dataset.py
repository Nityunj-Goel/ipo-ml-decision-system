from typing import Literal
import pandas as pd

redundant_cols = ['~Issue_Open_Date', 'closing_date']
percent_columns = ["ROCE", "ROE", "RoNW"]

def clean(df: pd.DataFrame, mode: Literal['inference', 'training'] = 'training') -> pd.DataFrame:
    df = df.copy()

    # Drop redundant columns (ignore if missing)
    df = df.drop(columns=redundant_cols, errors="ignore")

    # Clean percentage columns (only if present)
    p_cols = [c for c in percent_columns if c in df.columns]
    df[p_cols] = df[p_cols].apply(
        lambda s: pd.to_numeric(
            s.astype("string").str.replace("%", "", regex=False),
            errors="coerce" if mode == 'inference' else "raise"
        )
    )
    df.rename(columns={c: f"{c} (%)" for c in p_cols if not c.endswith("(%)")}, inplace=True)

    return df
