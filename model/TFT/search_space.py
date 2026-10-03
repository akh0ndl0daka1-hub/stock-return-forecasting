"""
TFT hyperparameter search space for Bayesian optimization (see bayes_opt.py).

Deliberately parallel to the LSTM space (LSTM/search_space.py) so the two
architectures are tuned on equal terms:

  - Same dimensionality: 7 searched dimensions each.
  - Three dimensions are IDENTICAL in name and support -- learning_rate,
    dropout, optimizer -- so the optimisation regime is not a confound when
    the two architectures are compared.
  - hidden_size mirrors the LSTM's `units` (64/128/256) and dense_units
    mirrors its `dense_units` (32/64/128), so representational capacity is
    searched over the same range.
  - num_layers mirrors the LSTM's stacked depth (1/2/3), here the depth of
    the TFT's recurrent encoder.
  - num_heads replaces the LSTM's `activation` dimension: it is the one
    architecture-specific degree of freedom, at the same cardinality (3)
    that keeps the two spaces comparably sized.

hidden_size is divisible by every value of num_heads, so the per-head
attention dimension is always integral.
"""

from __future__ import annotations
from skopt.space import Real, Categorical

TFT_SEARCH_SPACE = [
    Real(1e-4, 1e-3, prior="log-uniform", name="learning_rate"),
    Categorical([0.1, 0.2], name="dropout"),
    Categorical(["adam", "rmsprop"], name="optimizer"),
    Categorical([1, 2, 3], name="num_layers"),
    Categorical([64, 128, 256], name="hidden_size"),
    Categorical([32, 64, 128], name="dense_units"),
    Categorical([1, 2, 4], name="num_heads"),
]
