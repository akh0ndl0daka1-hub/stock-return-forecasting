"""LSTM hyperparameter search space for Bayesian optimization (see bayes_opt.py)."""

from __future__ import annotations
from skopt.space import Real, Categorical

LSTM_SEARCH_SPACE = [
    Categorical(["tanh", "relu"], name="activation"),
    Real(1e-4, 1e-3, prior="log-uniform", name="learning_rate"),
    Categorical([0.1, 0.2], name="dropout"),
    Categorical(["adam", "rmsprop"], name="optimizer"),
    Categorical([1, 2, 3], name="num_layers"),
    Categorical([64, 128, 256], name="units"),
    Categorical([32, 64, 128], name="dense_units"),
]
