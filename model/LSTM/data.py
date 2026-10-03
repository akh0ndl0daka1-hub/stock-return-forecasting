"""
Load a ticker's reduced feature file and split it into a fixed train
window and the walk-forward-eligible remainder.

Reads : Masters/feature_reduction/{TICKER}_reduced.csv
Target: {TICKER}_LOGR_Adj_Close   (daily log return; forced to column 0)
Price : {TICKER}_Adj_Close        (anchor for converting predicted log
                                    returns back to a price level)

TRAIN_END matches feature_reduction.py's own cutoff (2022-12-30): the same
date used there to keep both rolling periods' test years (2023, 2024) out
of any fitting decision. Everything up to and including TRAIN_END is the
walk-forward's fixed training window; 2023-01-01 onward is where
walk-forward folds are swept (covering both periods' test years in one
continuous sweep).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from run_paths import FEATURE_DIR, resolve_feature_dir

TRAIN_END = "2022-12-30"


def load_and_prepare(ticker: str, feature_dir: Path | None = None):
    """
    Returns:
        dates_array   : np.ndarray[datetime64], shape [N]
        data          : np.ndarray[float32], shape [N, F], target at column 0
        target_col    : str
        feature_cols  : list[str] (excludes target)
        prices_array  : np.ndarray[float32], shape [N] (adjusted close)
        length_train  : int, rows with Date <= TRAIN_END
        length_test   : int, rows with Date > TRAIN_END
    """
    feature_dir = resolve_feature_dir(feature_dir)
    path = feature_dir / f"{ticker}_reduced.csv"
    if not path.exists():
        raise FileNotFoundError(f"reduced feature file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{path}: dates are not monotonically increasing")
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicate dates found")

    target_col = f"{ticker}_LOGR_Adj_Close"
    close_col = f"{ticker}_Adj_Close"
    if target_col not in df.columns:
        raise KeyError(f"{path}: missing target column {target_col}")
    if close_col not in df.columns:
        raise KeyError(f"{path}: missing price column {close_col}")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        print(f"{n_missing} rows found")
        raise ValueError(f"{path}: {n_missing} missing values found; expected none")

    feature_cols = [c for c in df.columns if c != target_col]
    df_ordered = df[[target_col] + feature_cols]

    dates_array = df.index.to_numpy()
    data = df_ordered.to_numpy(dtype=np.float32)
    prices_array = df[close_col].to_numpy(dtype=np.float32)

    train_mask = df.index <= pd.Timestamp(TRAIN_END)
    length_train = int(train_mask.sum())
    length_test = len(df) - length_train

    return dates_array, data, target_col, feature_cols, prices_array, length_train, length_test
