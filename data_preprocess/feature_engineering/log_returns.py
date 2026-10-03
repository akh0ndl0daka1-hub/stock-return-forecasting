"""
Convert adjusted closing prices to daily log returns and document
stationarity, for every ticker.

For each ticker, reads the collected price file:
    <Masters>/data_collection/data/collected/tickers/prices_v/{TICKER}_prices_v.csv
    (column {TICKER}_Adj_Close)
computes the daily close-to-close log return of the adjusted close, and
writes:
    tickers_log_ret/{TICKER}_log_ret.csv
    (column {TICKER}_LOGR_Adj_Close)

Missing values are NOT removed: the first trading day of each ticker's
sample has no prior price to diff against, so its log return is NaN by
construction. That row is kept in the saved file rather than dropped.

Stationarity evidence
  Raw adjusted prices are non-stationary (unit root); their log returns are
  stationary. This is demonstrated with two complementary tests per series:
    * ADF  (Augmented Dickey-Fuller): H0 = a unit root is present
             (non-stationary). Rejecting H0 (small p) supports stationarity.
    * KPSS (Kwiatkowski-Phillips-Schmidt-Shin): H0 = the series IS stationary.
             Failing to reject H0 (large p) supports stationarity.
  The two have opposite null hypotheses, so agreement is strong evidence.
  Both tests require a NaN-free series, so the leading missing row is
  dropped only for the purpose of running these tests -- the saved log
  return file itself is unaffected. For each ticker the tests are run on
  BOTH the price and the log return, and a combined evidence table is
  written to:
    tickers_log_ret/stationary_evidence.csv

Run:
    python log_returns.py
    python log_returns.py --src ../../data_collection/data/collected/tickers/prices_v \
        --out tickers_log_ret
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTED_ROOT = REPO_ROOT / "data_collection" / "data" / "collected"

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

TICKERS = ["AAPL", "JNJ", "JPM", "NKE", "XOM", "BA"]


# --------------------------------------------------------------------------- #
# Loading and conversion
# --------------------------------------------------------------------------- #

def load_adj_close(path: Path, ticker: str) -> pd.Series | None:
    if not path.exists():
        print(f"[{ticker}] price file not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    col = f"{ticker}_Adj_Close"
    if col not in df.columns:
        raise KeyError(f"{col} not in {path}; have {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").dropna()


def to_log_returns(adj_close: pd.Series, ticker: str) -> pd.Series:
    """
    Daily close-to-close log return: r_t = ln(P_t / P_{t-1}).
    The first observation is NaN (no prior price) and is kept as-is.
    """
    r = np.log(adj_close / adj_close.shift(1))
    r.name = f"{ticker}_LOGR_Adj_Close"
    return r


# --------------------------------------------------------------------------- #
# Stationarity tests
# --------------------------------------------------------------------------- #

def run_adf(series: pd.Series) -> tuple[float, float]:
    """ADF: returns (statistic, p-value). H0 = unit root (non-stationary)."""
    stat, p, *_ = adfuller(series.values, autolag="AIC")
    return float(stat), float(p)


def run_kpss(series: pd.Series) -> tuple[float, float]:
    """KPSS (level): returns (statistic, p-value). H0 = stationary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # p-value clipped-range note
        stat, p, *_ = kpss(series.values, regression="c", nlags="auto")
    return float(stat), float(p)


def evidence_row(ticker: str, kind: str, series: pd.Series) -> dict:
    adf_stat, adf_p = run_adf(series)
    kpss_stat, kpss_p = run_kpss(series)
    # Verdict: ADF rejects unit root (p<0.05) AND KPSS does not reject
    # stationarity (p>0.05) => stationary. Opposite nulls, so agreement is
    # stronger evidence than either test alone.
    stationary = (adf_p < 0.05) and (kpss_p > 0.05)
    return {
        "ticker": ticker,
        "series": kind,
        "n": len(series),
        "ADF_stat": round(adf_stat, 3),
        "ADF_p": round(adf_p, 4),
        "KPSS_stat": round(kpss_stat, 3),
        "KPSS_p": round(kpss_p, 4),
        "stationary": stationary,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Daily log returns + stationarity.")
    ap.add_argument("--src", default=str(COLLECTED_ROOT / "tickers" / "prices_v"),
                    help="folder containing {TICKER}_prices_v.csv")
    ap.add_argument("--out", default=str(HERE / "tickers_log_ret"))
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    evidence: list[dict] = []

    for t in args.tickers:
        adj = load_adj_close(src / f"{t}_prices_v.csv", t)
        if adj is None:
            continue

        logr = to_log_returns(adj, t)

        dst = out / f"{t}_log_ret.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        logr.to_frame().to_csv(dst)
        print(f"[{t}] {len(logr):>5} rows ({int(logr.isna().sum())} missing) "
              f"-> {dst}")

        # Stationarity tests can't handle NaN, so drop the leading missing
        # row here only -- the saved file itself keeps it.
        evidence.append(evidence_row(t, "adj_close_price", adj))
        evidence.append(evidence_row(t, "log_return", logr.dropna()))

    if evidence:
        ev = pd.DataFrame(evidence)
        ev_path = out / "stationary_evidence.csv"
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        ev.to_csv(ev_path, index=False)
        print(f"\nstationarity evidence -> {ev_path}\n")
        with pd.option_context("display.width", 120,
                               "display.max_columns", None):
            print(ev.to_string(index=False))
        prices = ev[ev.series == "adj_close_price"]
        rets = ev[ev.series == "log_return"]
        print(f"\nprices stationary:      {int(prices.stationary.sum())}/{len(prices)}")
        print(f"log returns stationary: {int(rets.stationary.sum())}/{len(rets)}")


if __name__ == "__main__":
    main()
