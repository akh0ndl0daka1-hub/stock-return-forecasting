# Verification tests

The test suite is designed for a quick independent check of the forecasting pipeline. It uses synthetic data and does not require the research datasets, saved model weights, market-data downloads, or FinBERT downloads.

Run the complete suite from the repository root:

```bash
pytest -q
```

The tests cover the following methodological contracts:

- chronological walk-forward partitioning and the 2024 holdout definition;
- supervised-window indexing and separation between input and target dates;
- fitting of feature and target scalers on training observations only;
- correlation-based feature reduction using pre-evaluation observations only;
- preservation of the adjusted-log-return target, adjusted closing price, and AAPL sentiment feature during feature reduction;
- forward filling without backfilling future observations;
- absence of pre-publication sentiment values;
- causal EWT features through a future-perturbation check;
- adjusted log-return calculation;
- quantile-loss and QRisk calculations;
- PICP, PINAW, AIS, and quantile-crossing treatment in adjusted log-return space;
- zero-return benchmark construction;
- origin-level loss aggregation for Diebold--Mariano testing, Bartlett HAC bandwidth, and identical-forecast behaviour;
- Holm multiple-testing adjustment;
- Kupiec unconditional coverage testing at the one-day horizon;
- matching and sign convention for the AAPL sentiment comparison;
- bootstrap reproducibility and temporal-stability matching;
- gradient-attribution shape, non-negativity, normalisation, and outer-quantile target;
- LSTM, TFT, and KAN output shape `[B, H, 3]`;
- flattened KAN input dimension and causal TFT attention;
- valid lookback-window/forecast-horizon combinations and experiment counts;
- hyperparameter search-space bounds and Bayesian-optimisation evaluation budget; and
- forecast-level output schema and reconstruction of evaluation metrics from saved predictions.

TensorFlow-specific tests use tiny CPU forward passes only. If TensorFlow is absent, pytest reports these tests as skipped while the numerical, temporal, statistical, and file-contract tests still run.
