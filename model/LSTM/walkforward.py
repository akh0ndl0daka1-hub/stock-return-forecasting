"""
Rolling walk-forward cross-validation for sequence models.

Per fold k (0-indexed), on the time axis:

    [-------- train --------][-- val --][-- test --]
    train_start          train_end   val_end    test_end

    train_start = k * step
    train_end   = train_start + window
    val_start   = train_end
    val_end     = val_start + horizon
    test_start  = val_end
    test_end    = test_start + horizon

`window` is fixed (a rolling, not expanding, window: both ends advance by
`step` each fold). `val` and `test` are each exactly `horizon` rows long,
built as a single sequence (batch size 1) -- only `train` is expanded into
many overlapping (X, y) samples via `supervised_window`.

Scalers are fit ONLY on the fold's own training block, never on val/test,
so no fold leaks information across the train/val/test boundary.
"""

from __future__ import annotations
from typing import Iterator

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

# Same two rolling periods used everywhere else in this project (EWT,
# feature_reduction's TRAIN_END): period1 trains through 2022, tests 2023;
# period2 trains through 2023, tests 2024.
PERIODS = {
    "2023": ("2010-01-04", "2022-12-30", "2023-01-03", "2023-12-29"),
    "2024": ("2011-01-03", "2023-12-29", "2024-01-02", "2024-12-31"),
}


def supervised_window(window: np.ndarray, lookback: int, horizon: int):
    """
    window: [T, F] array, target at column 0.
    Returns (X, y): X [N, lookback, F], y [N, horizon], both derived from
    every valid sliding position within `window`.
    """
    n, f = window.shape
    n_samples = n - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(f"window too short: {n} rows for lookback={lookback}, horizon={horizon}")

    X = sliding_window_view(window, window_shape=(lookback, f))[: n_samples, 0]
    y = sliding_window_view(window[:, 0], window_shape=horizon)[lookback:lookback + n_samples]
    return X, y


def rolling_walk_forward(
    dates_array: np.ndarray,
    prices_array: np.ndarray,
    data: np.ndarray,
    lookback: int,
    horizon: int,
    window: int,
    step: int = 1,
) -> Iterator[dict]:
    n = len(data)
    k = 0
    while True:
        train_start = k * step
        train_end = train_start + window
        val_start = train_end
        val_end = val_start + horizon
        test_start = val_end
        test_end = test_start + horizon
        if test_end > n:
            break

        train_block = data[train_start:train_end]
        X_train_raw, y_train_raw = supervised_window(train_block, lookback, horizon)

        feature_scaler = MinMaxScaler()
        feature_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
        target_scaler = MinMaxScaler()
        target_scaler.fit(y_train_raw.reshape(-1, 1))

        def scale_X(X):
            shp = X.shape
            return feature_scaler.transform(X.reshape(-1, shp[-1])).reshape(shp)

        def scale_y(y):
            shp = y.shape
            return target_scaler.transform(y.reshape(-1, 1)).reshape(shp)

        X_train = scale_X(X_train_raw)
        y_train = scale_y(y_train_raw)

        X_val_raw = data[val_start - lookback: val_start]
        y_val_raw = data[val_start:val_end, 0]
        X_val = scale_X(X_val_raw.reshape(1, lookback, -1))
        y_val = scale_y(y_val_raw.reshape(1, horizon))

        X_test_raw = data[test_start - lookback: test_start]
        y_test_raw = data[test_start:test_end, 0]
        X_test = scale_X(X_test_raw.reshape(1, lookback, -1))
        y_test = scale_y(y_test_raw.reshape(1, horizon))

        yield {
            "k": k,
            "train_idx": (train_start, train_end),
            "val_idx": (val_start, val_end),
            "test_idx": (test_start, test_end),
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "y_train_raw": y_train_raw,
            "y_val_raw": y_val_raw.reshape(1, horizon),
            "y_test_raw": y_test_raw.reshape(1, horizon),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "last_price_test": prices_array[test_start - 1],
            "last_price_train": prices_array[train_end - 1],
            "test_prices": prices_array[test_start:test_end],
            "train_dates": dates_array[train_start:train_end],
            "test_dates": np.datetime_as_string(dates_array[test_start:test_end], unit="D"),
        }
        k += 1


def build_period_fold(
    dates_array: np.ndarray,
    prices_array: np.ndarray,
    data: np.ndarray,
    lookback: int,
    horizon: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict:
    """
    ONE fold per rolling period, trained ONCE (not retrained per test day):
      - Train on [train_start, train_end], holding out the last `horizon`
        rows of that span as a single validation window (same val
        construction as rolling_walk_forward, just one span instead of
        many).
      - Test by sweeping EVERY valid day across the WHOLE [test_start,
        test_end] span as a separate prediction window (via
        supervised_window, sliding by 1 day) -- using the scalers and
        weights fit ONCE on this period's own training data. The model
        itself is trained by the caller on X_train/y_train; this function
        only builds the data, it does not train.
    """
    def idx(d: str) -> int:
        return int(dates_array.searchsorted(np.datetime64(pd.Timestamp(d))))

    train_start_i = idx(train_start)
    train_end_i = idx(train_end) + 1     # +1: include train_end's own row if present
    test_start_i = idx(test_start)
    test_end_i = min(idx(test_end) + 1, len(data))

    val_target_start = train_end_i - horizon
    train_block_end = val_target_start
    if train_block_end - train_start_i < lookback + horizon:
        raise ValueError(f"training span too short for lookback={lookback}, horizon={horizon}")

    train_block = data[train_start_i:train_block_end]
    X_train_raw, y_train_raw = supervised_window(train_block, lookback, horizon)

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X_train_raw.reshape(-1, X_train_raw.shape[-1]))
    target_scaler = MinMaxScaler()
    target_scaler.fit(y_train_raw.reshape(-1, 1))

    def scale_X(X):
        shp = X.shape
        return feature_scaler.transform(X.reshape(-1, shp[-1])).reshape(shp)

    def scale_y(y):
        shp = y.shape
        return target_scaler.transform(y.reshape(-1, 1)).reshape(shp)

    X_train = scale_X(X_train_raw)
    y_train = scale_y(y_train_raw)

    X_val_raw = data[val_target_start - lookback: val_target_start]
    y_val_raw = data[val_target_start:train_end_i, 0]
    X_val = scale_X(X_val_raw.reshape(1, lookback, -1))
    y_val = scale_y(y_val_raw.reshape(1, horizon))

    # Sweep every valid test window across the whole test span, in one
    # vectorized batch -- no retraining between windows.
    test_context = data[test_start_i - lookback: test_end_i]
    X_test_raw, y_test_raw = supervised_window(test_context, lookback, horizon)
    X_test = scale_X(X_test_raw)
    y_test = scale_y(y_test_raw)

    n_test = len(X_test_raw)
    last_prices_test = prices_array[test_start_i - 1: test_start_i - 1 + n_test]
    test_dates = np.datetime_as_string(dates_array[test_start_i: test_start_i + n_test], unit="D")

    return {
        "train_idx": (train_start_i, train_block_end),
        "val_idx": (val_target_start, train_end_i),
        "test_idx": (test_start_i, test_end_i),
        "n_test_windows": n_test,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw.reshape(1, horizon),
        "y_test_raw": y_test_raw,                    # [n_test, horizon]
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "last_prices_test": last_prices_test,         # [n_test]
        "test_dates": test_dates,                     # [n_test]
    }


def period_walk_forward(
    dates_array: np.ndarray,
    prices_array: np.ndarray,
    data: np.ndarray,
    lookback: int,
    horizon: int,
    periods: dict = PERIODS,
) -> Iterator[dict]:
    """Yields exactly len(periods) folds (2, by default): one per rolling period."""
    for k, (name, (tr_s, tr_e, te_s, te_e)) in enumerate(periods.items()):
        fold = build_period_fold(dates_array, prices_array, data, lookback, horizon,
                                 tr_s, tr_e, te_s, te_e)
        fold["k"] = k
        fold["period"] = name
        yield fold
