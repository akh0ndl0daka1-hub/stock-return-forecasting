"""
Kolmogorov-Arnold Network for quantile regression: architecture, device
resolution, and the per-fold training routine.

Deliberately mirrors LSTM/lstm_model.py and TFT/tft_model.py so the three
architectures differ only in how the input window is encoded. Identical
across all three:
    - input contract   : [B, lookback, n_features]
    - output contract  : [B, horizon, n_quantiles], direct multi-horizon,
                         emitted in a single forward pass
    - loss             : make_quantile_loss_tf(quantiles)   (shared module)
    - decoder          : dense head with ReLU, then a linear layer
    - fixed settings   : L2 1e-4, batch 32, patience 8, best-weight restore
    - optimiser set    : adam / rmsprop, same learning-rate range
    - device handling  : same GPU resolution and CPU fallback

Search space tuned via Bayesian optimization (see search_space.py):
    learning_rate : 1e-4 .. 1e-3         (log-uniform)   [same as LSTM/TFT]
    dropout       : 0.1, 0.2                             [same as LSTM/TFT]
    optimizer     : adam, rmsprop                        [same as LSTM/TFT]
    num_layers    : 1, 2, 3              (KAN layers)    [same as LSTM/TFT]
    dense_units   : 32, 64, 128          (head width)    [same as LSTM/TFT]
    width         : 64, 128, 256         (KAN layer width)
    grid_size     : 3, 5, 10             (spline resolution, KAN-specific)

Implementation note. The KAN layer formulation follows efficient-KAN by
Huanqi Cao (https://github.com/Blealtan/efficient-kan), which is distributed
under the MIT License. This file adapts that formulation to TensorFlow/Keras and
integrates it with the quantile forecasting, optimisation, training, and
evaluation pipeline used here. The upstream copyright and MIT permission notice
are preserved in LICENSES/efficient-kan-MIT.txt and THIRD_PARTY_NOTICES.md.

The implementation represents each edge with a base branch plus a B-spline
branch and is retained as the architecture used for the forecasting experiments.
"""

from __future__ import annotations
import gc
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers, initializers

from shared import make_quantile_loss_tf

# Fixed architecture hyperparameters (not tuned) -- identical to LSTM and TFT.
L2_REG = 1e-4
BATCH_SIZE = 32
PATIENCE = 8

# Fixed KAN settings (not tuned).
SPLINE_ORDER = 3
GRID_RANGE = (-1.0, 2.0)   # inputs are min-max scaled on training data, so
                           # test values can fall outside [0, 1]; the grid is
                           # widened so those values stay inside the spline
                           # support instead of being extrapolated flat.


def _resolve_tf_device(device: str | None = None) -> str:
    device = (device or "cuda").lower()
    if device.startswith("cpu"):
        return "/CPU:0"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        return "/GPU:0"
    warnings.warn("No GPU visible to TensorFlow; falling back to CPU.", RuntimeWarning)
    return "/CPU:0"


def _make_rmsprop_optimizer(learning_rate: float):
    try:
        legacy = getattr(optimizers, "legacy", None)
        if legacy is not None:
            return legacy.RMSprop(learning_rate=learning_rate)
    except ImportError:
        pass
    return optimizers.RMSprop(learning_rate=learning_rate)


@tf.keras.utils.register_keras_serializable(package="kan")
class KANDense(layers.Layer):
    """
    One KAN layer: every input-output edge carries a learned univariate
    function, expressed as a base term plus a B-spline expansion.

    Parameter count is in_dim * units * (grid_size + spline_order + 1), so it
    grows with the flattened window length. That growth is a property of the
    architecture and is reported rather than worked around.
    """

    def __init__(self, units: int, grid_size: int = 5, spline_order: int = SPLINE_ORDER,
                 grid_range: tuple[float, float] = GRID_RANGE, l2_reg: float = L2_REG,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.l2_reg = float(l2_reg)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        k, g = self.spline_order, self.grid_size
        lo, hi = self.grid_range
        step = (hi - lo) / g

        # extended knot vector, length g + 2k + 1, shared across input units
        knots = (np.arange(-k, g + k + 1, dtype=np.float32) * step + lo)
        self._grid = tf.constant(np.tile(knots[None, :], (in_dim, 1)), dtype=tf.float32)

        n_basis = g + k
        self.spline_weight = self.add_weight(
            name="spline_weight", shape=(in_dim, self.units, n_basis),
            initializer=initializers.RandomNormal(stddev=0.1 / np.sqrt(in_dim)),
            regularizer=regularizers.l2(self.l2_reg), trainable=True)
        self.base_weight = self.add_weight(
            name="base_weight", shape=(in_dim, self.units),
            initializer=initializers.GlorotUniform(),
            regularizer=regularizers.l2(self.l2_reg), trainable=True)
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def _bspline_bases(self, x):
        """x: [B, in_dim] -> [B, in_dim, grid_size + spline_order] via de Boor."""
        xe = tf.expand_dims(x, -1)                 # [B, in, 1]
        g = tf.expand_dims(self._grid, 0)          # [1, in, n_knots]

        bases = tf.cast((xe >= g[..., :-1]) & (xe < g[..., 1:]), tf.float32)
        for order in range(1, self.spline_order + 1):
            left_num = xe - g[..., : -(order + 1)]
            left_den = g[..., order:-1] - g[..., : -(order + 1)]
            right_num = g[..., order + 1:] - xe
            right_den = g[..., order + 1:] - g[..., 1:-order]
            bases = (left_num / left_den) * bases[..., :-1] + \
                    (right_num / right_den) * bases[..., 1:]
        return bases

    def call(self, inputs):
        base = tf.matmul(tf.nn.silu(inputs), self.base_weight)
        bases = self._bspline_bases(inputs)
        spline = tf.einsum("bik,iok->bo", bases, self.spline_weight)
        return base + spline + self.bias

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.units,)

    def get_config(self):
        return {
            **super().get_config(),
            "units": self.units,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": list(self.grid_range),
            "l2_reg": self.l2_reg,
        }


def build_kan_model(lookback: int, n_features: int, horizon: int,
                    quantiles: list[float], params: dict) -> tf.keras.Model:
    q_count = len(quantiles)
    num_layers = int(params["num_layers"])
    width = int(params["width"])
    dense_units = int(params["dense_units"])
    grid_size = int(params["grid_size"])
    dropout = float(params["dropout"])

    inputs = layers.Input(shape=(lookback, n_features), name="inputs")

    # The complete lookback window is flattened into distinct time-feature
    # coordinates before entering the conventional KAN. Positions therefore
    # remain separate input coordinates, but the explicit sequential structure
    # is not retained after flattening. This is distinct from temporal KAN
    # variants that add sequence-specific mechanisms.
    x = layers.Flatten(name="flatten_window")(inputs)

    for i in range(num_layers):
        x = KANDense(width, grid_size=grid_size, name=f"kan_{i + 1}")(x)
        # activations leaving a KAN layer are unbounded, while the spline grid
        # is finite; normalising returns them to the grid's support before the
        # next layer.
        x = layers.LayerNormalization(name=f"kan_{i + 1}_norm")(x)
        x = layers.Dropout(dropout, name=f"kan_{i + 1}_drop")(x)

    # Same decoder as the LSTM and TFT, so the three differ only in encoding.
    head = layers.Dense(dense_units, activation="relu",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        name="dense_head")(x)
    raw_out = layers.Dense(horizon * q_count, activation="linear",
                           name="raw_quantile_output")(head)
    outputs = layers.Reshape((horizon, q_count), name="quantile_output")(raw_out)

    model = models.Model(inputs, outputs, name="kan_quantile_model")

    lr = float(params["learning_rate"])
    optimizer_name = params["optimizer"]
    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        opt = _make_rmsprop_optimizer(lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if hasattr(opt, "jit_compile"):
        opt.jit_compile = False

    model.compile(optimizer=opt, loss=make_quantile_loss_tf(quantiles), jit_compile=False)
    return model


_GPU_FALLBACK_SUBSTRINGS = (
    "JIT compilation failed", "Op:Pow", "adam/Pow", "rmsprop/Sqrt",
    "Dnn is not supported", "CudnnRNN", "OOM when allocating",
)


def train_kan_model(X_train, y_train, X_val, y_val, lookback, horizon, n_features,
                    quantiles, params, max_epochs=100, tf_device="/GPU:0"):
    """Signature-identical to train_lstm_model and train_tft_model."""
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    def _fit(device):
        model = build_kan_model(lookback, n_features, horizon, quantiles, params)
        cb_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=int(params.get("patience", PATIENCE)),
            restore_best_weights=True, verbose=0)
        with tf.device(device):
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=int(max_epochs), batch_size=int(params.get("batch_size", BATCH_SIZE)),
                callbacks=[cb_early], verbose=0)
        return model, history.history

    try:
        return _fit(tf_device)
    except (tf.errors.UnknownError, tf.errors.InvalidArgumentError,
            tf.errors.ResourceExhaustedError) as e:
        msg = str(e)
        if tf_device == "/GPU:0" and any(s in msg for s in _GPU_FALLBACK_SUBSTRINGS):
            warnings.warn(f"GPU training failed ({msg[:120]}...); retrying on CPU.", RuntimeWarning)
            tf.keras.backend.clear_session()
            gc.collect()
            return _fit("/CPU:0")
        raise
