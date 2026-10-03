from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model" / "LSTM"))

from interval_metrics import order_interval_bounds, compute_interval_metrics


def test_crossed_bounds_are_recorded_then_ordered():
    low = np.array([0.0, 2.0])
    high = np.array([1.0, 1.0])
    ordered_low, ordered_high, rate = order_interval_bounds(low, high)
    assert rate == 0.5
    assert np.allclose(ordered_low, [0.0, 1.0])
    assert np.allclose(ordered_high, [1.0, 2.0])


def test_interval_metrics_use_supplied_return_scale():
    actual = np.array([-0.01, 0.00, 0.01])
    lower = np.array([-0.02, -0.01, 0.00])
    upper = np.array([0.00, 0.01, 0.02])
    out = compute_interval_metrics(actual, lower, upper, alpha=0.2)
    assert out["picp"] == 1.0
    assert np.isclose(out["ais"], 0.02)
