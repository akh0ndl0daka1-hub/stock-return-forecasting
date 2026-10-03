"""Bayesian hyperparameter optimisation shared by LSTM, TFT, and KAN.

Bayesian optimisation uses a Gaussian-process surrogate with a fixed evaluation budget
for every forecasting architecture.  ``gp_minimize`` supplies that surrogate;
``gp_hedge`` adaptively chooses among expected improvement, probability of
improvement, and lower-confidence-bound acquisition rules.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any

from skopt import gp_minimize


@dataclass
class BayesOptResult:
    """Compact result object consumed by the architecture runners."""

    best_params: dict[str, Any]
    best_value: float
    trials: list[dict[str, Any]]
    result: Any


def run_bayes_opt(
    objective: Callable[..., float],
    search_space: Iterable,
    *,
    n_calls: int = 20,
    n_initial_points: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> BayesOptResult:
    """Minimise ``objective`` with a Gaussian-process surrogate.

    The default search evaluates 20 candidates, including five initial
    search points, with random state 42 controlling candidate generation.  The random state governs Bayesian
    optimisation only; it does not make neural-network fitting deterministic.
    """
    dimensions = list(search_space)
    names = [d.name for d in dimensions]
    if any(name is None for name in names):
        raise ValueError("Every search-space dimension must have a name.")
    if n_calls < n_initial_points:
        raise ValueError("n_calls must be >= n_initial_points")

    def wrapped(values):
        params = dict(zip(names, values))
        return float(objective(**params))

    result = gp_minimize(
        wrapped,
        dimensions,
        n_calls=int(n_calls),
        n_initial_points=int(n_initial_points),
        acq_func="gp_hedge",
        random_state=int(random_state),
        verbose=bool(verbose),
    )

    trials = []
    for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals), start=1):
        row = {name: value for name, value in zip(names, x)}
        row.update({"trial": i, "value": float(y)})
        trials.append(row)

    return BayesOptResult(
        best_params={name: value for name, value in zip(names, result.x)},
        best_value=float(result.fun),
        trials=trials,
        result=result,
    )
