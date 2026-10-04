from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAPL", "BA", "JNJ", "JPM", "NKE", "XOM"]
LOOKBACKS = [5, 20, 60]
HORIZONS = [1, 5, 20]
EXPECTED = [(5, 1), (20, 1), (20, 5), (60, 1), (60, 5), (60, 20)]


def test_valid_lookback_horizon_combinations():
    combos = [(lb, h) for lb in LOOKBACKS for h in HORIZONS if lb > h]
    assert combos == EXPECTED


def test_experiment_grid_count():
    combos = [(t, lb, h) for t in TICKERS for lb in LOOKBACKS for h in HORIZONS if lb > h]
    assert len(combos) == 36
    assert len(combos) * 3 == 108
    assert len(combos) * 3 * 2 == 216


def _literal_lists(path: Path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"Categorical", "Real"}:
            name = None
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name = kw.value.value
            if name:
                if node.func.id == "Categorical" and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                    values[name] = [ast.literal_eval(e) for e in node.args[0].elts]
                elif node.func.id == "Real":
                    values[name] = (ast.literal_eval(node.args[0]), ast.literal_eval(node.args[1]))
    return values


def test_bayesian_search_space_bounds():
    lstm = _literal_lists(ROOT / "model" / "LSTM" / "search_space.py")
    tft = _literal_lists(ROOT / "model" / "TFT" / "search_space.py")
    kan = _literal_lists(ROOT / "model" / "KAN" / "search_space.py")
    for space in (lstm, tft, kan):
        assert space["learning_rate"] == (1e-4, 1e-3)
        assert space["dropout"] == [0.1, 0.2]
        assert space["optimizer"] == ["adam", "rmsprop"]
        assert space["num_layers"] == [1, 2, 3]
        assert space["dense_units"] == [32, 64, 128]
    assert lstm["units"] == [64, 128, 256]
    assert tft["hidden_size"] == [64, 128, 256]
    assert tft["num_heads"] == [1, 2, 4]
    assert kan["width"] == [64, 128, 256]
    assert kan["grid_size"] == [3, 5, 10]


def test_bayesian_candidate_budget_and_seed(monkeypatch):
    # Load bayes_opt.py with a tiny fake skopt module, so this contract test has
    # no dependency on scikit-optimize or expensive optimisation.
    captured = {}
    fake = ModuleType("skopt")

    def fake_gp_minimize(func, dimensions, **kwargs):
        captured.update(kwargs)
        x_iters = [[0.1], [0.2], [0.3]]
        vals = [func(x) for x in x_iters]
        best = int(np.argmin(vals))
        return SimpleNamespace(x_iters=x_iters, func_vals=np.array(vals), x=x_iters[best], fun=vals[best])

    fake.gp_minimize = fake_gp_minimize
    monkeypatch.setitem(sys.modules, "skopt", fake)
    bo = load_from_path("test_bayes_opt_contract", ROOT / "model" / "bayes_opt.py")

    dim = SimpleNamespace(name="x")
    result = bo.run_bayes_opt(lambda x: x, [dim], n_calls=20, n_initial_points=5, random_state=42)
    assert captured["n_calls"] == 20
    assert captured["n_initial_points"] == 5
    assert captured["random_state"] == 42
    assert captured["acq_func"] == "gp_hedge"
    assert result.best_params == {"x": 0.1}
