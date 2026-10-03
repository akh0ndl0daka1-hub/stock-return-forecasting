"""
Gradient-based (vanilla saliency) feature importance, computed per fold and
saved as ONE file covering every fold and every feature -- no per-fold
files, no top-N truncation.

Method: |d(target)/d(input)| via tf.GradientTape on the fold's test
sequence, target = equal-weight sum of the low and high quantile predictions across
the forecast horizon. This measures sensitivity of the combined outer-quantile
output and is not interpreted as prediction-interval width. Absolute gradients
are averaged over evaluation observations and lookback positions to obtain one
importance value per feature per temporal period.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def compute_fold_importance(model, X_test: np.ndarray, low_idx: int, high_idx: int,
                            tf_device: str) -> np.ndarray | None:
    """Returns [n_features] importance for one fold's test sequence, or None if no gradient."""
    x_explain = tf.convert_to_tensor(X_test.astype(np.float32))
    with tf.device(tf_device):
        with tf.GradientTape() as tape:
            tape.watch(x_explain)
            pred = model(x_explain, training=False)   # [B, H, Q]
            target_score = 0.5 * (
                tf.reduce_sum(pred[:, :, low_idx]) + tf.reduce_sum(pred[:, :, high_idx])
            )
        grads = tape.gradient(target_score, x_explain)

    if grads is None:
        return None
    grad_abs = np.abs(grads.numpy())          # [B, lookback, n_features]
    feature_time_imp = grad_abs.mean(axis=0)  # [lookback, n_features]
    return feature_time_imp.mean(axis=0)      # [n_features]


def save_feature_importance_across_folds(
    importances_by_fold: dict[int, np.ndarray],
    feature_names: list[str],
    out_path: Path,
) -> pd.DataFrame:
    """
    Writes one CSV: rows = every feature (none dropped), columns = one per
    fold plus `mean` and `std` across folds. Sorted by mean importance
    descending for readability, but nothing is truncated.
    """
    df = pd.DataFrame(importances_by_fold, index=feature_names)
    df.columns = [f"fold_{k}" for k in df.columns]
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1)
    df = df.sort_values("mean", ascending=False)
    df.index.name = "feature"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df
