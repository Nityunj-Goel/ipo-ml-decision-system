from collections.abc import Iterable
import numpy as np
from src.utils.utils import load_config


def compute_allocation(
    probabilities: Iterable[float],
    t_min: float,
    alpha: float,
) -> np.ndarray:
    """Compute portfolio allocation weights from predicted probabilities.

    Args:
        probabilities: 1-D iterable of predicted probabilities that each IPO's
            listing gain will exceed the configured threshold, each in [0, 1].
        t_min: Minimum probability threshold for inclusion, in [0, 1].
        alpha: Concentration parameter. Higher alpha concentrates weight
            on higher-probability IPOs.

    Returns:
        Array of allocation weights (same length as probabilities).
        Weights sum to <= 1. Entries below threshold are 0.
        If no IPO passes the threshold, all weights are 0 (hold cash).
    """
    probs = np.asarray(list(probabilities), dtype=float)
    max_allocation = load_config()["portfolio"]["max_allocation"]

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1-D iterable")
    if not np.all((probs >= 0) & (probs <= 1)):
        raise ValueError("all probabilities must be in [0, 1]")
    if not (0 <= t_min <= 1):
        raise ValueError("t_min must be in [0, 1]")

    raw = np.where(probs >= t_min, probs ** alpha, 0.0)
    total = raw.sum()
    if total == 0:
        return np.zeros_like(probs)

    weights = raw / total
    weights = np.minimum(weights, max_allocation)
    return weights
