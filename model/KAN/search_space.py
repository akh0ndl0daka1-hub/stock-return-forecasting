"""
KAN hyperparameter search space for Bayesian optimization (see bayes_opt.py).

Parallel to the LSTM and TFT spaces so the three architectures are tuned on
equal terms:

  - Same dimensionality: 7 searched dimensions.
  - Five dimensions are IDENTICAL in name and support to the LSTM's --
    learning_rate, dropout, optimizer, num_layers, dense_units -- so the
    optimisation regime is not a confound in the comparison.
  - width takes the same values as the LSTM's `units` and the TFT's
    `hidden_size`, so encoder capacity is searched over the same range.
  - grid_size is the one architecture-specific dimension, replacing the
    LSTM's `activation` and the TFT's `num_heads`. It controls the
    resolution of the learned univariate functions, which is the parameter
    with no counterpart in the other two architectures.

Categorical grid size is 324, the same as the TFT's and larger than the
LSTM's 216, for the same reason: the architecture-specific dimension has
three values where the LSTM's activation has two.
"""

from __future__ import annotations
from skopt.space import Real, Categorical

KAN_SEARCH_SPACE = [
    Real(1e-4, 1e-3, prior="log-uniform", name="learning_rate"),
    Categorical([0.1, 0.2], name="dropout"),
    Categorical(["adam", "rmsprop"], name="optimizer"),
    Categorical([1, 2, 3], name="num_layers"),
    Categorical([64, 128, 256], name="width"),
    Categorical([32, 64, 128], name="dense_units"),
    Categorical([3, 5, 10], name="grid_size"),
]
