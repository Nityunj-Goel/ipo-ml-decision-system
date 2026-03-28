from os import PathLike
from pathlib import Path
import pandas as pd

date_cols = ['ipoStartDate', 'ipoEndDate', '~IPO_Listing_Date']

def find_project_root(start: Path = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("Project root not found")

def read_dataframe_from_csv(path: PathLike) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=date_cols, date_format="%Y-%m-%dT%H:%M:%S.%fZ")

def save_dataframe_to_csv(df: pd.DataFrame, path: PathLike, *args, **kwargs):
    kwargs['index'] = False
    df.to_csv(path, *args, **kwargs)