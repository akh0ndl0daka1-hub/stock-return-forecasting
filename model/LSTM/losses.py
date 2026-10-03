"""
Quantile (pinball) loss and QRisk -- ported verbatim (same formulas, same
input-shape handling) from the original ProjectX LSTM pipeline's
training/losses.py, so results are directly comparable.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import tensorflow as tf


def make_quantile_loss_tf(quantiles: List[float]):
    """
    TensorFlow/Keras pinball (quantile) loss for predictions of shape [B, H, Q].

    Expected input shapes
    ---------------------
    y_true: [B], [B, H], or [B, H, 1]
    y_pred: [B, H, Q]
    """
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    qs = tf.constant(quantiles, dtype=tf.float32)

    def _loss_fn(y_true, y_pred):
        y_true_t = tf.cast(y_true, tf.float32)
        y_pred_t = tf.cast(y_pred, tf.float32)

        if y_true_t.shape.rank == 1:
            y_true_t = tf.reshape(y_true_t, (-1, 1))
        if y_true_t.shape.rank == 2:
            y_true_t = tf.expand_dims(y_true_t, axis=-1)

        if y_pred_t.shape.rank is not None and y_pred_t.shape.rank != 3:
            raise ValueError(f"y_pred must have rank 3 [B, H, Q], got rank {y_pred_t.shape.rank}")

        q = tf.reshape(qs, (1, 1, -1))  # [1, 1, Q]
        err = y_true_t - y_pred_t
        loss = tf.maximum(q * err, (q - 1.0) * err)
        return tf.reduce_mean(loss)

    return _loss_fn


def qrisk(y_true, y_pred_all_q, quantiles: List[float]) -> Dict[float, float]:
    """
    y_true       : [N] or [N, H]
    y_pred_all_q : [N, Q] (H=1) or [N, H, Q]
    Returns {quantile: QRisk}, QRisk_q = 2 * sum(pinball_q) / sum(|y_true|).
    """
    if not quantiles:
        raise ValueError("quantiles must not be empty")

    qs = np.asarray(quantiles, dtype=np.float64)

    y = np.asarray(y_true, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    elif y.ndim != 2:
        raise ValueError(f"y_true must have rank 1 or 2, got shape {y.shape}")

    yhat = np.asarray(y_pred_all_q, dtype=np.float64)

    if yhat.ndim == 2:
        n_samples, q_count = yhat.shape
        if q_count != len(quantiles):
            raise ValueError(f"y_pred_all_q last dim = {q_count}, expected {len(quantiles)}")
        if y.shape != (n_samples, 1):
            raise ValueError(f"For rank-2 y_pred_all_q, expected y_true shape {(n_samples, 1)}, got {y.shape}")
        yhat = yhat.reshape(n_samples, 1, q_count)
    elif yhat.ndim == 3:
        n_samples, horizon, q_count = yhat.shape
        if q_count != len(quantiles):
            raise ValueError(f"y_pred_all_q last dim = {q_count}, expected {len(quantiles)}")
        if y.shape != (n_samples, horizon):
            raise ValueError(f"y_true shape {y.shape} does not match expected {(n_samples, horizon)}")
    else:
        raise ValueError(f"y_pred_all_q must have rank 2 or 3, got shape {yhat.shape}")

    den = float(np.abs(y).sum(dtype=np.float64))
    if den < 1e-12:
        den = 1e-12

    def _pinball(y_val: np.ndarray, yhat_val: np.ndarray, q_val: float) -> np.ndarray:
        e = y_val - yhat_val
        return np.maximum(q_val * e, (q_val - 1.0) * e)

    q_risks: Dict[float, float] = {}
    for qi, q in enumerate(qs):
        yhat_q = yhat[..., qi]
        num = float(_pinball(y, yhat_q, float(q)).sum(dtype=np.float64))
        q_risks[float(q)] = float(2.0 * num / den)

    return q_risks
