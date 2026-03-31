import pandas as pd
from os import PathLike
from pathlib import Path

def find_project_root(start: Path = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("Project root not found")

def save_dataframe_to_csv(df: pd.DataFrame, path: PathLike, *args, **kwargs):
    kwargs['index'] = False
    df.to_csv(path, *args, **kwargs)