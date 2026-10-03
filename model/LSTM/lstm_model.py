"""
LSTM quantile-regression model: architecture, device resolution, and the
per-fold training routine.

Search space tuned via Bayesian optimization (see search_space.py):
    activation    : tanh, relu           (LSTM cell activation)
    learning_rate : 1e-4 .. 1e-3         (log-uniform)
    dropout       : 0.1, 0.2
    optimizer     : adam, rmsprop
    num_layers    : 1, 2, 3              (stacked LSTM layers)
    units         : 64, 128, 256         (LSTM layer width)
    dense_units   : 32, 64, 128          (dense head width)

Everything else (L2 regularization, batch size, early-stopping patience)
is held fixed at a sensible default -- not part of this search, since it
wasn't requested.
"""

from __future__ import annotations
import gc
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

from losses import make_quantile_loss_tf

# Fixed architecture hyperparameters (not tuned).
L2_REG = 1e-4
BATCH_SIZE = 32
PATIENCE = 8


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
    # tf.keras.optimizers.legacy is present as a module in Keras 3 but
    # raises ImportError on instantiation unless the standalone tf_keras
    # package is installed; just use the standard optimizer directly.
    try:
        legacy = getattr(optimizers, "legacy", None)
        if legacy is not None:
            return legacy.RMSprop(learning_rate=learning_rate)
    except ImportError:
        pass
    return optimizers.RMSprop(learning_rate=learning_rate)


def build_lstm_model(lookback: int, n_features: int, horizon: int,
                     quantiles: list[float], params: dict) -> tf.keras.Model:
    q_count = len(quantiles)
    num_layers = int(params["num_layers"])
    units = int(params["units"])
    dense_units = int(params["dense_units"])
    inputs = layers.Input(shape=(lookback, n_features), name="inputs")

    x = inputs
    for i in range(num_layers):
        x = layers.LSTM(
            units=units,
            activation=params["activation"],
            return_sequences=(i < num_layers - 1),   # only the last layer collapses the sequence
            dropout=float(params["dropout"]),
            recurrent_dropout=0.0,
            recurrent_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(L2_REG),
            use_cudnn=False,
            name=f"lstm_{i + 1}",
        )(x)

    x = layers.Dense(dense_units, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_REG), name="dense_head")(x)
    raw_out = layers.Dense(horizon * q_count, activation="linear", name="raw_quantile_output")(x)
    outputs = layers.Reshape((horizon, q_count), name="quantile_output")(raw_out)

    model = models.Model(inputs, outputs, name="lstm_quantile_model")

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
    "Dnn is not supported", "CudnnRNN",
)


def train_lstm_model(X_train, y_train, X_val, y_val, lookback, horizon, n_features,
                     quantiles, params, max_epochs=100, tf_device="/GPU:0"):
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    def _fit(device):
        model = build_lstm_model(lookback, n_features, horizon, quantiles, params)
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
    except (tf.errors.UnknownError, tf.errors.InvalidArgumentError) as e:
        msg = str(e)
        if tf_device == "/GPU:0" and any(s in msg for s in _GPU_FALLBACK_SUBSTRINGS):
            warnings.warn(f"GPU training failed ({msg[:120]}...); retrying on CPU.", RuntimeWarning)
            tf.keras.backend.clear_session()
            gc.collect()
            return _fit("/CPU:0")
        raise
