from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]


def _load_tf_architecture(module_name: str, folder: str, filename: str):
    tf = pytest.importorskip("tensorflow")
    arch_dir = ROOT / "model" / folder
    # Architecture modules import `shared` by a local name.  Put their own folder first.
    old = list(sys.path)
    for name in ["shared", "search_space", module_name]:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(arch_dir))
    sys.path.insert(1, str(ROOT / "model"))
    sys.path.insert(2, str(ROOT / "model" / "LSTM"))
    try:
        mod = load_from_path(module_name, arch_dir / filename)
    finally:
        sys.path[:] = old
    return tf, mod


def test_feature_attribution_shape_and_nonnegative():
    tf = pytest.importorskip("tensorflow")
    feat = load_from_path("test_feature_importance_tf", ROOT / "model" / "LSTM" / "feature_importance.py")
    inputs = tf.keras.Input(shape=(3, 2))
    flat = tf.keras.layers.Flatten()(inputs)
    raw = tf.keras.layers.Dense(2 * 3, use_bias=False, kernel_initializer="ones")(flat)
    outputs = tf.keras.layers.Reshape((2, 3))(raw)
    model = tf.keras.Model(inputs, outputs)
    X = np.ones((4, 3, 2), dtype=np.float32)
    imp = feat.compute_fold_importance(model, X, low_idx=0, high_idx=2, tf_device="/CPU:0")
    assert imp.shape == (2,)
    assert np.all(imp >= 0)
    shares = imp / imp.sum()
    assert np.isclose(shares.sum(), 1.0)


def test_outer_quantile_attribution_ignores_median_only_path():
    tf = pytest.importorskip("tensorflow")
    feat = load_from_path("test_feature_importance_target_tf", ROOT / "model" / "LSTM" / "feature_importance.py")
    # q0.1=x0, q0.5=1000*x1, q0.9=x0.  Correct outer-quantile attribution
    # should therefore be sensitive to feature 0 and insensitive to feature 1.
    inp = tf.keras.Input(shape=(1, 2))

    x0 = tf.keras.layers.Lambda(lambda x: x[:, :, 0:1])(inp)
    x1 = tf.keras.layers.Lambda(lambda x: 1000.0 * x[:, :, 1:2])(inp)
    out = tf.keras.layers.Concatenate(axis=-1)([x0, x1, x0])
    model = tf.keras.Model(inputs=inp, outputs=out)
    X = np.array([[[1.0, 1.0]], [[2.0, 3.0]]], dtype=np.float32)
    imp = feat.compute_fold_importance(model, X, 0, 2, "/CPU:0")
    assert imp[0] > 0
    assert np.isclose(imp[1], 0.0, atol=1e-12)


@pytest.mark.parametrize("folder,filename,builder_name,params", [
    ("LSTM", "lstm_model.py", "build_lstm_model", {
        "num_layers": 1, "units": 8, "dense_units": 4, "activation": "tanh",
        "dropout": 0.1, "optimizer": "adam", "learning_rate": 1e-3}),
    ("TFT", "tft_model.py", "build_tft_model", {
        "num_layers": 1, "hidden_size": 8, "dense_units": 4, "num_heads": 2,
        "dropout": 0.1, "optimizer": "adam", "learning_rate": 1e-3}),
    ("KAN", "kan_model.py", "build_kan_model", {
        "num_layers": 1, "width": 8, "dense_units": 4, "grid_size": 3,
        "dropout": 0.1, "optimizer": "adam", "learning_rate": 1e-3}),
])
def test_model_output_shapes(folder, filename, builder_name, params):
    tf, mod = _load_tf_architecture(f"shape_{folder.lower()}_model", folder, filename)
    model = getattr(mod, builder_name)(lookback=5, n_features=3, horizon=2,
                                      quantiles=[0.1, 0.5, 0.9], params=params)
    y = model(np.zeros((2, 5, 3), dtype=np.float32), training=False)
    assert tuple(y.shape) == (2, 2, 3)


def test_kan_flattened_input_dimension():
    _, mod = _load_tf_architecture("flatten_kan_model", "KAN", "kan_model.py")
    params = {"num_layers": 1, "width": 8, "dense_units": 4, "grid_size": 3,
              "dropout": 0.1, "optimizer": "adam", "learning_rate": 1e-3}
    model = mod.build_kan_model(lookback=5, n_features=3, horizon=2,
                                quantiles=[0.1, 0.5, 0.9], params=params)
    layer = model.get_layer("flatten_window")
    assert int(layer.output.shape[-1]) == 15


def test_tft_attention_is_causal():
    tf, mod = _load_tf_architecture("causal_tft_model", "TFT", "tft_model.py")
    tf.random.set_seed(7)
    inp = tf.keras.Input(shape=(4, 8))
    out = mod._interpretable_attention(inp, hidden=8, num_heads=2, dropout=0.0, name="causal_test")
    model = tf.keras.Model(inp, out)
    x = np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 10.0
    y1 = model(x, training=False).numpy()
    changed = x.copy()
    changed[:, 2:, :] += 1000.0  # alter only positions later than t=1
    y2 = model(changed, training=False).numpy()
    assert np.allclose(y1[:, :2, :], y2[:, :2, :], atol=1e-5, rtol=1e-5)
