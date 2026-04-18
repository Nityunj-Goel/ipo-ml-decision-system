"""
Inference pipeline: chains a trained prediction pipeline with capital allocation.

Given a fitted sklearn Pipeline exposing predict_proba and a DataFrame of
features (with internal column names), produces per-IPO positive-class
probabilities and allocation weights. Input IPOs are assumed to be for a
single day, since allocation weights are normalized across the batch.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.portfolio.allocator import compute_allocation


class InferencePipeline:
    """prediction_pipeline -> probability -> capital allocator -> % allocation per IPO."""

    def __init__(self, fitted_prediction_pipeline: Pipeline, t_min: float, alpha: float):
        check_is_fitted(fitted_prediction_pipeline)
        self._pipeline = fitted_prediction_pipeline
        self.t_min = t_min
        self.alpha = alpha
        self._pos_idx = list(fitted_prediction_pipeline.classes_).index(1)

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        probabilities = self._pipeline.predict_proba(X)[:, self._pos_idx]
        weights = compute_allocation(probabilities, t_min=self.t_min, alpha=self.alpha, normalize=True)
        return probabilities, weights
