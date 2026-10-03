"""
Temporal Fusion Transformer (Lim et al., 2021) for quantile regression:
architecture, device resolution, and the per-fold training routine.

Deliberately mirrors LSTM/lstm_model.py so the two architectures differ only
in the encoder. Identical across both:
    - input contract   : [B, lookback, n_features]
    - output contract  : [B, horizon, n_quantiles], direct multi-horizon,
                         emitted in a single forward pass (no recursion)
    - loss             : make_quantile_loss_tf(quantiles)   (shared module)
    - fixed settings   : L2 1e-4, batch 32, patience 8, best-weight restore
    - optimiser set    : adam / rmsprop, same learning-rate range
    - device handling  : same GPU resolution and CPU fallback

Search space tuned via Bayesian optimization (see search_space.py):
    learning_rate : 1e-4 .. 1e-3         (log-uniform)   [same as LSTM]
    dropout       : 0.1, 0.2                             [same as LSTM]
    optimizer     : adam, rmsprop                        [same as LSTM]
    num_layers    : 1, 2, 3              (recurrent encoder depth)
    hidden_size   : 64, 128, 256         (model width, d_model)
    dense_units   : 32, 64, 128          (output head width)
    num_heads     : 1, 2, 4              (attention heads)

Architecture, following Lim et al. Since this dataset has no static
covariates and no known-future inputs (every feature is an observed past
input), the static covariate encoders and the future-input decoder of the
original are not instantiated; what remains is the observed-input path:

    input -> variable selection network
          -> recurrent encoder (locality enhancement)
          -> gate + skip + norm
          -> enrichment GRN
          -> interpretable multi-head self-attention (causally masked)
          -> gate + skip + norm
          -> position-wise GRN
          -> gate + skip + norm
          -> last position -> dense head -> [horizon, quantiles]

The attention is the interpretable variant of Lim et al.: per-head query and
key projections with a value projection SHARED across heads, head outputs
averaged rather than concatenated, then a single output projection.
"""

from __future__ import annotations
import gc
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

from shared import make_quantile_loss_tf

# Fixed architecture hyperparameters (not tuned) -- identical to the LSTM.
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
    # Same handling as the LSTM: tf.keras.optimizers.legacy exists as a module
    # in Keras 3 but raises on instantiation without the standalone tf_keras
    # package, so fall through to the standard optimizer.
    try:
        legacy = getattr(optimizers, "legacy", None)
        if legacy is not None:
            return legacy.RMSprop(learning_rate=learning_rate)
    except ImportError:
        pass
    return optimizers.RMSprop(learning_rate=learning_rate)


@tf.keras.utils.register_keras_serializable(package="tft")
class WeightedSum(layers.Layer):
    """
    Variable-selection combination step: contracts the per-variable feature
    axis of [B, T, F, H] against selection weights [B, T, F, 1], giving
    [B, T, H]. Registered so saved models reload without custom_objects.
    """

    def __init__(self, axis: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        transformed, weights = inputs
        return tf.reduce_sum(transformed * weights, axis=self.axis)

    def compute_output_shape(self, input_shape):
        shp = list(input_shape[0])
        del shp[self.axis]
        return tuple(shp)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis}


def _gated_residual_network(x, units: int, dropout: float, name: str,
                            output_units: int | None = None):
    """GRN: dense -> ELU -> dense -> dropout -> GLU -> add skip -> layer norm."""
    output_units = output_units or units

    skip = x
    if x.shape[-1] != output_units:
        skip = layers.Dense(output_units, kernel_regularizer=regularizers.l2(L2_REG),
                            name=f"{name}_skip_proj")(x)

    h = layers.Dense(units, kernel_regularizer=regularizers.l2(L2_REG),
                     name=f"{name}_dense_1")(x)
    h = layers.Activation("elu", name=f"{name}_elu")(h)
    h = layers.Dense(units, kernel_regularizer=regularizers.l2(L2_REG),
                     name=f"{name}_dense_2")(h)
    h = layers.Dropout(dropout, name=f"{name}_drop")(h)

    return _gate_add_norm(h, skip, output_units, name=f"{name}_gan")


def _gate_add_norm(x, skip, units: int, name: str):
    """GLU on x, residual add with skip, then layer normalisation."""
    gate = layers.Dense(units, activation="sigmoid",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        name=f"{name}_gate")(x)
    lin = layers.Dense(units, kernel_regularizer=regularizers.l2(L2_REG),
                       name=f"{name}_lin")(x)
    gated = layers.Multiply(name=f"{name}_glu")([gate, lin])
    added = layers.Add(name=f"{name}_add")([skip, gated])
    return layers.LayerNormalization(name=f"{name}_norm")(added)


def _variable_selection(inputs, lookback: int, n_features: int, hidden: int,
                        dropout: float, name: str = "vsn"):
    """
    Per-timestep variable selection over observed inputs.

    Each scalar variable gets its own projection into `hidden` dimensions
    (EinsumDense supplies independent weights per variable), and a GRN over
    the flattened input produces a softmax weight per variable per timestep.
    The weighted sum collapses the variable axis.
    """
    # [B, L, F] -> [B, L, F, 1] -> per-variable projection [B, L, F, hidden]
    expanded = layers.Reshape((lookback, n_features, 1), name=f"{name}_expand")(inputs)
    transformed = layers.EinsumDense(
        "blfi,fio->blfo",
        output_shape=(lookback, n_features, hidden),
        bias_axes="fo",
        name=f"{name}_var_proj",
    )(expanded)

    # selection weights over variables, conditioned on the whole timestep
    weights = _gated_residual_network(inputs, hidden, dropout,
                                      name=f"{name}_weight_grn",
                                      output_units=n_features)
    weights = layers.Activation("softmax", name=f"{name}_softmax")(weights)
    weights = layers.Reshape((lookback, n_features, 1), name=f"{name}_w_expand")(weights)

    return WeightedSum(axis=2, name=f"{name}_combine")([transformed, weights])


def _interpretable_attention(x, hidden: int, num_heads: int, dropout: float,
                             name: str = "attn"):
    """
    Interpretable multi-head self-attention (Lim et al.): per-head query and
    key projections, a value projection shared across heads, head outputs
    averaged rather than concatenated, then one output projection. Causally
    masked so position t attends only to positions <= t.
    """
    d_attn = max(1, hidden // num_heads)

    value = layers.Dense(d_attn, kernel_regularizer=regularizers.l2(L2_REG),
                         name=f"{name}_value_shared")(x)

    heads = []
    for h in range(num_heads):
        q = layers.Dense(d_attn, kernel_regularizer=regularizers.l2(L2_REG),
                         name=f"{name}_q_{h}")(x)
        k = layers.Dense(d_attn, kernel_regularizer=regularizers.l2(L2_REG),
                         name=f"{name}_k_{h}")(x)
        heads.append(
            layers.Attention(use_scale=True, dropout=dropout,
                             name=f"{name}_head_{h}")([q, value, k],
                                                      use_causal_mask=True)
        )

    merged = heads[0] if num_heads == 1 else layers.Average(name=f"{name}_head_avg")(heads)
    return layers.Dense(hidden, kernel_regularizer=regularizers.l2(L2_REG),
                        name=f"{name}_out_proj")(merged)


def build_tft_model(lookback: int, n_features: int, horizon: int,
                    quantiles: list[float], params: dict) -> tf.keras.Model:
    q_count = len(quantiles)
    num_layers = int(params["num_layers"])
    hidden = int(params["hidden_size"])
    dense_units = int(params["dense_units"])
    num_heads = int(params["num_heads"])
    dropout = float(params["dropout"])

    inputs = layers.Input(shape=(lookback, n_features), name="inputs")

    # 1. variable selection over observed inputs
    selected = _variable_selection(inputs, lookback, n_features, hidden, dropout)

    # 2. recurrent encoder (locality enhancement), depth searched like the LSTM's
    x = selected
    for i in range(num_layers):
        x = layers.LSTM(
            units=hidden,
            return_sequences=True,            # attention consumes the full sequence
            dropout=dropout,
            recurrent_dropout=0.0,
            recurrent_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l2(L2_REG),
            use_cudnn=False,
            name=f"encoder_lstm_{i + 1}",
        )(x)

    # 3. gate + skip over the encoder, then enrichment
    encoded = _gate_add_norm(x, selected, hidden, name="encoder_gan")
    enriched = _gated_residual_network(encoded, hidden, dropout, name="enrichment")

    # 4. causally-masked interpretable self-attention, gated back onto enrichment
    attended = _interpretable_attention(enriched, hidden, num_heads, dropout)
    attended = _gate_add_norm(attended, enriched, hidden, name="attn_gan")

    # 5. position-wise feed-forward, gated back onto the encoder output
    ff = _gated_residual_network(attended, hidden, dropout, name="pos_ff")
    ff = _gate_add_norm(ff, encoded, hidden, name="pos_ff_out")

    # 6. last position -> dense head -> [horizon, quantiles]
    last = layers.Cropping1D(cropping=(lookback - 1, 0), name="last_step")(ff)
    last = layers.Reshape((hidden,), name="last_step_flat")(last)

    head = layers.Dense(dense_units, activation="relu",
                        kernel_regularizer=regularizers.l2(L2_REG),
                        name="dense_head")(last)
    raw_out = layers.Dense(horizon * q_count, activation="linear",
                           name="raw_quantile_output")(head)
    outputs = layers.Reshape((horizon, q_count), name="quantile_output")(raw_out)

    model = models.Model(inputs, outputs, name="tft_quantile_model")

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


def train_tft_model(X_train, y_train, X_val, y_val, lookback, horizon, n_features,
                    quantiles, params, max_epochs=100, tf_device="/GPU:0"):
    """Signature-identical to train_lstm_model, so callers are interchangeable."""
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)

    def _fit(device):
        model = build_tft_model(lookback, n_features, horizon, quantiles, params)
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
