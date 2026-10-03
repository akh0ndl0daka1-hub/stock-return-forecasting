# Data availability

Research datasets are not distributed with this repository. The forecasting pipeline combines market data, company valuation data, macroeconomic and cross-asset series, and AAPL textual information obtained from multiple sources. Some of these sources may restrict redistribution.

After preprocessing and temporal alignment, the modelling sample contains 3,774 trading-day observations from 4 January 2010 to 31 December 2024. The correlation-based reduction step produces 49 modelling columns for AAPL, 52 for JNJ, 54 for JPM, 51 for NKE, 58 for XOM, and 54 for BA, excluding the CSV date index.

To run the forecasting code, prepare the reduced modelling files in `feature_reduction/` using the filenames documented in the root `README.md`. The sentiment-inclusive AAPL file must contain `AAPL_sentiment_score`.

Users are responsible for obtaining the source datasets from their original providers and complying with the applicable access and redistribution terms.
