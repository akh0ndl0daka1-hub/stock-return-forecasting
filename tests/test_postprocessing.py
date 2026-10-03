from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reproduce_results", ROOT / "model" / "reproduce_results.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    p = np.array([0.01, 0.04, 0.03])
    adj = rr.holm_adjust(p)
    assert np.all((adj >= p) & (adj <= 1))
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_dm_identical_losses_has_zero_difference():
    x = np.linspace(0.01, 0.02, 50)
    mean_d, stat, p = rr.dm_test(x, x, horizon=1)
    assert mean_d == 0.0
    assert stat == 0.0
    assert p == 1.0
