import pandas as pd
import yaml
from os import PathLike
from pathlib import Path

def get_project_root(start: Path = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent

    raise RuntimeError("Project root not found")

def load_config() -> dict:
    root = get_project_root()
    with open(root / "configs" / "config.yml") as f:
        return yaml.safe_load(f)


def load_raw_dataset() -> pd.DataFrame:
    root = get_project_root()
    return pd.read_csv(root / "data" / "aggregated" / "dataset.csv")


def save_dataframe_to_csv(df: pd.DataFrame, path: PathLike, *args, **kwargs):
    kwargs['index'] = False
    df.to_csv(path, *args, **kwargs)