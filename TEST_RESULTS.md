# Test execution report

The verification suite was executed in the packaging environment with:

```bash
pytest -q
```

Result:

```text
34 passed, 10 skipped
```

The skipped tests are the TensorFlow-dependent model and gradient smoke tests. TensorFlow is not installed in the packaging environment. They are not test failures; pytest will execute them automatically in an environment created from `requirements.txt`, which includes TensorFlow.

The non-TensorFlow suite completed in under one second and covered temporal ordering, training-only scaling, feature reduction, forward filling, causal EWT perturbation, adjusted log returns, interval metrics, statistical procedures, experiment-grid construction, Bayesian-optimisation call settings, output schemas, and reconstruction from saved predictions.

In addition, all Python files were checked with `python -m compileall -q .` without syntax errors.
