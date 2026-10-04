# Stock-return forecasting with multilevel financial information and textual sentiment

Code accompanying the MSc Computer Science thesis **“Integrating AI-driven Textual and Time Series Analysis to Enhance Precision in Stock Price Prediction”** (Rhodes University, 2026).

The repository compares three forecasting architectures: a long short-term memory network (LSTM), an observed-input temporal fusion transformer (TFT), and a Kolmogorov–Arnold network (KAN). Each model estimates the 0.1, 0.5, and 0.9 conditional quantiles of future daily adjusted log returns. The principal experiment covers six US-listed companies, lookback windows of 5, 20, and 60 trading days, and forecast horizons of 1, 5, and 20 trading days, retaining combinations for which the lookback window exceeds the forecast horizon. A separate AAPL experiment evaluates the addition of FinBERT-derived sentiment to the same numerical feature set.

## Experimental design

The workflow preserves temporal ordering throughout model development and evaluation. Hyperparameters are selected with Bayesian optimisation using a Gaussian-process surrogate, 20 candidate evaluations, five initial points, and random state 42 for candidate generation. Each candidate is evaluated over two development periods. Final model weights are fitted separately for the 2023 and 2024 evaluation periods, with 2024 retained as the final temporal holdout.

All three forecasting architectures use the same conditional-quantile objective and produce outputs with shape `[B, H, 3]`, where `H` is the forecast horizon. Evaluation includes median-forecast MAE, QRisk, prediction interval coverage probability (PICP), prediction interval normalised average width (PINAW), average interval score (AIS), zero-return benchmark comparisons, statistical tests, temporal stability, feature attribution, and computational time.

## Environment

The experiments were run with Python 3.10.12. Create an environment and install the required packages with:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick verification

The repository includes an examiner-facing unit-test suite built from small synthetic examples. It checks temporal ordering, training-only scaling, feature reduction, forward filling, causal EWT construction, adjusted log returns, quantile and interval calculations, statistical procedures, experiment-grid construction, saved-output schemas, feature attribution, and model output contracts. Run:

```bash
pytest -q
```

The tests do not download market data or FinBERT weights. TensorFlow-specific model and gradient smoke tests run when TensorFlow is installed and are skipped otherwise. See `TESTING.md` for the full test map.

The exact package-version snapshot from the original long-running experiments was not preserved. `requirements.txt` therefore lists the required packages without artificial version pins. For a new replication, record the resolved environment after installation, for example:

```bash
pip freeze > environment-lock.txt
```

## Data preparation

The repository contains scripts for market-data collection, adjusted log-return construction, technical indicators, empirical wavelet transform features, FinBERT sentiment, missing-value handling, data merging, and correlation-based feature reduction.

Final modelling files are expected at:

```text
feature_reduction/AAPL_reduced.csv
feature_reduction/BA_reduced.csv
feature_reduction/JNJ_reduced.csv
feature_reduction/JPM_reduced.csv
feature_reduction/NKE_reduced.csv
feature_reduction/XOM_reduced.csv
```

The sentiment-inclusive AAPL file must contain `AAPL_sentiment_score`. After correlation-based reduction, the modelling datasets contain 49 columns for AAPL, 52 for JNJ, 54 for JPM, 51 for NKE, 58 for XOM, and 54 for BA, excluding the CSV date index.

Create the numerical-only AAPL input with:

```bash
python feature_reduction_no_sentiment/prepare_aapl_no_sentiment.py
```

Research datasets are not distributed in this repository. Several inputs originate from third-party sources and may be subject to redistribution restrictions. See `data/README.md`.

## Running the experiment

Run all 36 valid company/lookback/horizon configurations for one architecture with:

```bash
python model/LSTM/main.py
python model/KAN/main.py
python model/TFT/main.py
```

Run an individual configuration directly, for example:

```bash
python model/LSTM/run_lstm.py --ticker AAPL --lookback 20 --horizon 5
```

For the independently optimised numerical-only AAPL comparison:

```bash
python model/LSTM/main.py --tickers AAPL --feature-dir feature_reduction_no_sentiment
python model/KAN/main.py  --tickers AAPL --feature-dir feature_reduction_no_sentiment
python model/TFT/main.py  --tickers AAPL --feature-dir feature_reduction_no_sentiment
```

The default output trees are `model/<ARCH>/results/` and `model/<ARCH>/results_no_sentiment/`.

## Saved outputs

Each forecasting configuration retains:

- `bayes_opt_trials.csv`: evaluated hyperparameters and validation objective;
- `train_results.csv`: period-specific training metrics;
- `test_results.csv`: period-specific MAE, QRisk, PICP, PINAW, AIS, and quantile-crossing rate;
- `test_predictions.csv`: forecast origin, forecast step, target date, realised adjusted log return, and predicted quantiles;
- `feature_importance_across_folds.csv`: period-specific gradient attribution;
- `summary.json`: selected hyperparameters and computational timing; and
- `models/*.keras`: fitted period-specific models.

PICP, PINAW, and AIS are calculated directly in adjusted log-return space. Quantile crossing is measured from the original 0.1 and 0.9 forecasts; the two outer quantiles are ordered only when constructing the interval used for PICP, PINAW, and AIS.

Adjusted closing-price paths are reconstructed separately for descriptive outputs and are not used to calculate prediction-interval metrics.

## Statistical post-processing

After the forecasting runs complete, reconstruct the comparative and statistical analyses with:

```bash
python model/reproduce_results.py --verify-reference
```

The script produces horizon-level summaries, zero-return benchmark comparisons, pairwise architecture comparisons, Diebold–Mariano tests using origin-level mean absolute loss, one-day Kupiec coverage tests, Holm-adjusted p-values, the paired AAPL sentiment analysis, temporal-stability summaries, feature-family attribution summaries, hyperparameter summaries, and computational-time summaries.

`reference/reported_results.json` stores selected aggregate results from the completed experiments so that a new reconstruction can be checked against the archived outputs.

## Reproducibility

Random state 42 controls Bayesian-optimisation candidate generation and the AAPL bootstrap analysis. Neural-network initialisation and training were not globally seeded in the original experiment, so a complete retraining run can produce small numerical differences even when the same data and candidate sequence are used. The retained forecast-level outputs provide the most direct basis for reconstructing the reported statistical analyses without refitting the neural networks.

## KAN implementation notice

The KAN layer formulation in `model/KAN/kan_model.py` follows the efficient-KAN implementation by Huanqi Cao (`Blealtan/efficient-kan`). The upstream project uses the MIT License. The implementation here is adapted to TensorFlow/Keras and integrated with the forecasting pipeline. The upstream copyright and licence notice are retained in `LICENSES/efficient-kan-MIT.txt`; further attribution details are provided in `THIRD_PARTY_NOTICES.md`.

## Repository notes

`IMPLEMENTATION_NOTES.md` documents the preparation of the standalone repository, including restored modules, output conventions, and known reproducibility constraints.
