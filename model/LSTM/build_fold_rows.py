"""Flatten one forecast origin into horizon-step records for reproducible analysis."""

from __future__ import annotations
import numpy as np


def build_fold_rows(
    fold_k: int,
    split: str,
    log_returns_pred: np.ndarray,
    log_returns_actual: np.ndarray,
    prices_pred: np.ndarray,
    prices_actual: np.ndarray,
    quantiles: list[float],
    dates: np.ndarray,
    lookback: int,
    horizon: int,
    *,
    period: str | None = None,
    forecast_origin: str | None = None,
) -> list[dict]:
    """Return one row per forecast step.

    ``TargetDate`` is the date whose adjusted log return is forecast, while
    ``ForecastOrigin`` is the last observed date available before the first
    target in that multi-step forecast. Price columns are retained only for
    descriptive reconstruction; statistical interval evaluation uses the
    adjusted-log-return columns.
    """
    q_idx = {q: i for i, q in enumerate(quantiles)}
    low_q, high_q = min(quantiles), max(quantiles)
    rows = []
    for h in range(horizon):
        raw_low = float(log_returns_pred[h, q_idx[low_q]])
        raw_high = float(log_returns_pred[h, q_idx[high_q]])
        row = {
            "Fold": fold_k,
            "Period": period,
            "Split": split,
            "ForecastOrigin": forecast_origin,
            "ForecastStep": h + 1,
            "TargetDate": str(dates[h]),
            "Date": str(dates[h]),  # retained for backward-compatible plotting
            "Lookback": lookback,
            "Horizon": horizon,
            "Actual_LogReturn": float(log_returns_actual[h]),
            "Actual_Price": float(prices_actual[h]),
            "QuantileCrossing": bool(raw_low > raw_high),
            "Pred_Price_PI_Width": float(abs(
                prices_pred[h, q_idx[high_q]] - prices_pred[h, q_idx[low_q]]
            )),
        }
        for q in quantiles:
            row[f"Pred_LogReturn_Q{q:.2f}"] = float(log_returns_pred[h, q_idx[q]])
            row[f"Pred_Price_Q{q:.2f}"] = float(prices_pred[h, q_idx[q]])
        rows.append(row)
    return rows
