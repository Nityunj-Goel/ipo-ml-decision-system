import joblib
import pandas as pd
import yaml
from os import PathLike
from pathlib import Path
from sklearn.pipeline import Pipeline

from configs.feature_config import RAW_FEATURES


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


def load_raw_dataset(from_year: int | None = None) -> pd.DataFrame:
    root = get_project_root()
    cfg = load_config()
    df = pd.read_csv(root / cfg["paths"]["raw_dataset"])
    if from_year is not None:
        year = RAW_FEATURES["year"]
        df = df[df[year] >= from_year].reset_index(drop=True)
    return df


def save_dataframe_to_csv(df: pd.DataFrame, path: PathLike, *args, **kwargs):
    kwargs['index'] = False
    df.to_csv(path, *args, **kwargs)


def save_pipeline(pipeline: Pipeline, path: PathLike | None = None):
    """Serialize a trained pipeline to disk so the API can load it."""
    if path is None:
        cfg = load_config()
        path = get_project_root() / cfg["paths"]["model"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)