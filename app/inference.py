import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from configs.feature_config import RAW_FEATURES
from src.portfolio.allocator import compute_allocation
from src.utils.utils import load_config

# Maps clean API field names → internal column names expected by the trained pipeline.
_FIELD_TO_COLUMN = {
    "nii": RAW_FEATURES["nii"],
    "qib": RAW_FEATURES["qib"],
    "retail": RAW_FEATURES["retail"],
    "total": RAW_FEATURES["total"],
    "year": RAW_FEATURES["year"],
    "issue_amount": RAW_FEATURES["issue_amount"],
    "price_band_high": RAW_FEATURES["price_band_high"],
    "price_band_low": RAW_FEATURES["price_band_low"],
    "gmp": RAW_FEATURES["gmp"],
}


class InferencePipeline:
    """Trained prediction pipeline + capital allocation logic.

    Loaded once at application startup.  Call :meth:`predict` with a list
    of IPO dicts (matching the API schema field names) to get back
    predicted probabilities and portfolio allocation weights.
    """

    def __init__(
        self,
        model_path: Path,
        t_min: float | None = None,
        alpha: float | None = None,
    ):
        self._pipeline: Pipeline = joblib.load(model_path)
        portfolio_cfg = load_config()["portfolio"]
        self.t_min = t_min if t_min is not None else portfolio_cfg["trade_threshold"]
        self.alpha = alpha if alpha is not None else portfolio_cfg["alpha"]
        self._pos_idx = list(self._pipeline.classes_).index(1)

    def predict(self, ipos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities, allocation_weights) arrays aligned to input order."""
        df = self._to_dataframe(ipos)
        probabilities = self._pipeline.predict_proba(df)[:, self._pos_idx]
        weights = compute_allocation(probabilities, t_min=self.t_min, alpha=self.alpha)
        return probabilities, weights

    @staticmethod
    def _to_dataframe(ipos: list[dict]) -> pd.DataFrame:
        records = [
            {_FIELD_TO_COLUMN[field]: value for field, value in ipo.items() if field in _FIELD_TO_COLUMN}
            for ipo in ipos
        ]
        return pd.DataFrame(records)
