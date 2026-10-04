from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]
interval = load_from_path("test_interval_module", ROOT / "model" / "LSTM" / "interval_metrics.py")


def _load_losses_with_tf():
    pytest.importorskip("tensorflow")
    return load_from_path("test_losses_tf_module", ROOT / "model" / "LSTM" / "losses.py")


def test_quantile_loss_known_values():
    tf = pytest.importorskip("tensorflow")
    losses_mod = _load_losses_with_tf()
    fn = losses_mod.make_quantile_loss_tf([0.1, 0.5, 0.9])
    y_true = tf.constant([[1.0]], dtype=tf.float32)
    y_pred = tf.constant([[[0.0, 0.5, 2.0]]], dtype=tf.float32)
    # pinball losses: 0.1*(1)=0.1; 0.5*(0.5)=0.25; (0.9-1)*(-1)=0.1
    expected = (0.1 + 0.25 + 0.1) / 3.0
    assert np.isclose(float(fn(y_true, y_pred).numpy()), expected)


def test_quantile_loss_output_is_scalar():
    tf = pytest.importorskip("tensorflow")
    losses_mod = _load_losses_with_tf()
    fn = losses_mod.make_quantile_loss_tf([0.1, 0.5, 0.9])
    y_true = tf.zeros((2, 4), dtype=tf.float32)
    y_pred = tf.zeros((2, 4, 3), dtype=tf.float32)
    out = fn(y_true, y_pred)
    assert out.shape.rank == 0


def test_interval_metrics_known_example():
    actual = np.array([-1.0, 0.0, 2.0])
    lower = np.array([-2.0, -0.5, 0.0])
    upper = np.array([0.0, 0.5, 1.0])
    out = interval.compute_interval_metrics(actual, lower, upper, alpha=0.2)
    assert np.isclose(out["picp"], 2 / 3)
    expected_pinaw = np.mean([2.0, 1.0, 1.0]) / 3.0
    assert np.isclose(out["pinaw"], expected_pinaw)
    # widths 2,1,1; final point is 1 above upper -> +10 penalty.
    assert np.isclose(out["ais"], (2 + 1 + 11) / 3)


def test_interval_metrics_use_log_returns_not_price_columns(rr):
    base = pd.DataFrame({
        "Actual_LogReturn": [-0.01, 0.0, 0.01],
        "Pred_LogReturn_Q0.10": [-0.02, -0.01, 0.0],
        "Pred_LogReturn_Q0.50": [-0.01, 0.001, 0.009],
        "Pred_LogReturn_Q0.90": [0.0, 0.01, 0.02],
        "Actual_Price": [100, 1000, 1_000_000],
        "Pred_Price_Q0.10": [1, 1, 1],
        "Pred_Price_Q0.90": [1e9, 1e9, 1e9],
    })
    first = rr.interval_metrics_from_rows(base)
    changed = base.copy()
    changed[["Actual_Price", "Pred_Price_Q0.10", "Pred_Price_Q0.90"]] *= -9999
    second = rr.interval_metrics_from_rows(changed)
    assert first == second


def test_quantile_crossing_recorded_before_reordering(rr):
    low_raw = np.array([0.0, 2.0])
    high_raw = np.array([1.0, 1.0])
    lo, hi, rate = interval.order_interval_bounds(low_raw, high_raw)
    assert rate == 0.5
    assert np.array_equal(lo, [0.0, 1.0])
    assert np.array_equal(hi, [1.0, 2.0])

    df = pd.DataFrame({
        "Actual_LogReturn": [0.5, 1.5],
        "Pred_LogReturn_Q0.10": low_raw,
        "Pred_LogReturn_Q0.50": [0.5, 1.5],
        "Pred_LogReturn_Q0.90": high_raw,
    })
    out = rr.interval_metrics_from_rows(df)
    assert out["crossing_rate"] == 0.5
    assert out["picp"] == 1.0


def test_zero_return_benchmark_mae(rr):
    y = np.array([-0.02, 0.01, 0.03])
    df = pd.DataFrame({
        "Actual_LogReturn": y,
        "Pred_LogReturn_Q0.10": y - 0.01,
        "Pred_LogReturn_Q0.50": [0.0, 0.0, 0.0],
        "Pred_LogReturn_Q0.90": y + 0.01,
    })
    out = rr.interval_metrics_from_rows(df)
    assert np.isclose(out["zero_mae"], np.mean(np.abs(y)))
    assert np.isclose(out["mae"], out["zero_mae"])
