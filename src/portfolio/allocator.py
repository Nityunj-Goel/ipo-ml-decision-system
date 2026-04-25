from collections.abc import Iterable
import numpy as np


def compute_allocation(
    probabilities: Iterable[float],
    t_min: float,
    normalize: bool = True,
) -> np.ndarray:
    """Compute portfolio allocation weights from predicted probabilities.

    All IPOs whose probability meets ``t_min`` receive an equal weight; the
    rest receive zero. No per-IPO cap is applied.

    Args:
        probabilities: 1-D iterable of predicted probabilities that each IPO's
            listing gain will exceed the configured threshold, each in [0, 1].
        t_min: Minimum probability threshold for inclusion, in [0, 1].
        normalize: If True, equal weights sum to 1 across selected IPOs. If
            False, each selected IPO receives a weight of 1.

    Returns:
        Array of allocation weights (same length as probabilities). Entries
        below threshold are 0. If no IPO passes the threshold, all weights
        are 0 (hold cash).
    """
    probs = np.asarray(list(probabilities), dtype=float)

    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1-D iterable")
    if not np.all((probs >= 0) & (probs <= 1)):
        raise ValueError("all probabilities must be in [0, 1]")
    if not (0 <= t_min <= 1):
        raise ValueError("t_min must be in [0, 1]")

    selected = probs >= t_min
    n_selected = int(selected.sum())
    if n_selected == 0:
        return np.zeros_like(probs)

    weight = 1.0 / n_selected if normalize else 1.0
    return np.where(selected, weight, 0.0)
