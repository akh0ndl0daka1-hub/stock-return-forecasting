"""Prediction-interval quality metrics in adjusted log-return space."""

from __future__ import annotations
import numpy as np


def order_interval_bounds(lower: np.ndarray, upper: np.ndarray):
    """Order outer quantiles for interval scoring while retaining crossing rate."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    crossing = lower > upper
    return np.minimum(lower, upper), np.maximum(lower, upper), float(np.mean(crossing))


def compute_interval_metrics(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                              alpha: float) -> dict[str, float]:
    """Calculate PICP, PINAW, and AIS from already ordered interval bounds."""
    actual = np.asarray(actual, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()

    if not (actual.size == lower.size == upper.size):
        raise ValueError("actual, lower, and upper must contain the same number of values")
    if np.any(lower > upper):
        raise ValueError("interval bounds are crossed; order them before metric calculation")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    inside = (actual >= lower) & (actual <= upper)
    picp = float(np.mean(inside))

    rng = float(actual.max() - actual.min()) if actual.size else 0.0
    pinaw = float(np.mean(upper - lower) / rng) if rng > 0 else 0.0

    width = upper - lower
    below = actual < lower
    above = actual > upper
    scores = width.copy()
    scores[below] += (2.0 / alpha) * (lower[below] - actual[below])
    scores[above] += (2.0 / alpha) * (actual[above] - upper[above])

    return {"picp": picp, "pinaw": pinaw, "ais": float(np.mean(scores))}
