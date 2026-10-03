"""Convert a sequence of predicted log returns back to a price level."""

from __future__ import annotations
import numpy as np


def logreturn_to_price(log_returns: np.ndarray, initial_price: float) -> np.ndarray:
    """
    log_returns: [H] or [H, Q]. Cumulative compounding from `initial_price`:
    price[0] = initial_price * exp(log_returns[0]), price[i] = price[i-1] * exp(log_returns[i]).
    """
    log_returns = np.asarray(log_returns, dtype=np.float64)
    cum = np.cumsum(log_returns, axis=0)
    return initial_price * np.exp(cum)
