"""
App-layer adapter: translates API request dicts to the DataFrame format
expected by the core InferencePipeline, then delegates to it.
"""
import numpy as np
import pandas as pd

from configs.feature_config import RAW_FEATURES
from src.pipelines.inference_pipeline import InferencePipeline

# Maps API field names -> internal column names the trained pipeline expects.
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


class InferenceService:
    """Adapts API request payloads to the core InferencePipeline."""

    def __init__(self, pipeline: InferencePipeline):
        self._pipeline = pipeline

    def predict(self, ipos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        df = self._to_dataframe(ipos)
        return self._pipeline.predict(df)

    @staticmethod
    def _to_dataframe(ipos: list[dict]) -> pd.DataFrame:
        records = [
            {_FIELD_TO_COLUMN[field]: value for field, value in ipo.items() if field in _FIELD_TO_COLUMN}
            for ipo in ipos
        ]
        return pd.DataFrame(records)
