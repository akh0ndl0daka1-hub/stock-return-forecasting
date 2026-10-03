# Implementation notes

This repository was prepared from the research code archive used for the forecasting experiments. The original archive contained the principal model and preprocessing code but was not self-contained for a clean public release. The preparation work below makes the execution path explicit and removes transient files and research data from the software package.

## Repository preparation

The following components were added or revised:

- `model/bayes_opt.py` provides the common Bayesian-optimisation wrapper used by the three forecasting architectures.
- `model/run_paths.py` resolves feature directories and separates the standard and numerical-only AAPL result trees.
- `model/reproduce_results.py` reconstructs comparative and statistical analyses from saved forecast-level outputs.
- `feature_reduction_no_sentiment/prepare_aapl_no_sentiment.py` creates the numerical-only AAPL feature set by removing only `AAPL_sentiment_score` from the reduced sentiment-inclusive dataset.
- The architecture entry points enumerate the complete set of six valid lookback-window/forecast-horizon combinations for all six companies.
- Forecast-level output records retain the temporal period, forecast origin, forecast step, target date, realised adjusted log return, and the 0.1, 0.5, and 0.9 forecasts.
- Evaluation MAE, prediction-interval metrics, and quantile-crossing rates are retained in period-specific result files.
- Prediction-interval metrics are calculated in adjusted log-return space. Original outer-quantile crossings are recorded before the two interval bounds are ordered for interval scoring.
- Research datasets, fitted model files, caches, logs, and operating-system metadata are excluded from the public package.

## Feature-reduction check

The correlation-based feature-reduction procedure was rerun on the supplied merged datasets during repository preparation. The resulting modelling-column counts, excluding the CSV date index, were:

| Company | Columns after reduction |
| --- | ---: |
| AAPL | 49 |
| JNJ | 52 |
| JPM | 54 |
| NKE | 51 |
| XOM | 58 |
| BA | 54 |

The regenerated AAPL output contains `AAPL_sentiment_score`. An older reduced AAPL file in the original archive omitted that column and is not included here.

## Statistical reconstruction

`model/reproduce_results.py` operates on the saved forecast-level and configuration-level outputs. It includes:

- zero-return benchmark calculations;
- Diebold–Mariano tests based on mean absolute error across the forecast steps belonging to each forecast origin;
- Bartlett HAC variance with bandwidth `H - 1` and the finite-sample adjustment for multi-step forecasts;
- Holm adjustment within defined comparison families;
- one-day Kupiec unconditional-coverage tests;
- paired AAPL Wilcoxon tests and bootstrap intervals;
- temporal-stability summaries;
- architecture count summaries;
- feature-family attribution aggregation;
- selected-hyperparameter summaries; and
- runtime summaries.

## Reproducibility constraints

Bayesian-optimisation candidate generation uses random state 42. The AAPL bootstrap also uses seed 42. Neural-network fitting was not globally seeded, so exact numerical replication of a complete retraining run is not guaranteed. Saved forecast-level outputs should be retained alongside any archival release when exact reconstruction of the statistical analyses is required.

## KAN provenance

The KAN layer formulation in `model/KAN/kan_model.py` follows the efficient-KAN implementation by Huanqi Cao. The upstream project is MIT-licensed, and the implementation here adapts the formulation to TensorFlow/Keras for quantile forecasting. The required upstream copyright and permission notice are retained in `LICENSES/efficient-kan-MIT.txt`.
