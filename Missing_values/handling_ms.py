"""
Handle missing values across the whole Masters pipeline by aligning every
source onto ONE canonical trading-day calendar: AAPL's own trading days
(verified identical across all 6 tickers and all 6 sector ETFs), restricted
to the common window 2010-01-04..2024-12-31.

Every category below is aligned the same way: reindex onto the union of
the canonical calendar and the file's own dates, forward-fill across that
union (so a value that lands on a non-trading day still carries into the
next trading day rather than being silently dropped), then narrow back
down to the canonical calendar. This one operation happens to resolve
several different problems at once:
    - valuation ratios / AAPL news: quarterly / per-article -> daily
    - log returns: drops the single leading NaN (2009-01-02, before the
      window)
    - technical indicators: drops the 60-row lookback warm-up (also
      entirely before the window)
    - copper: the only source that does NOT already share AAPL's exact
      calendar -- it lacks 3 trading days AAPL has (2016-10-10,
      2016-11-11, 2018-01-29) and has 1 day AAPL doesn't (2023-11-23,
      likely a COMEX/NYSE holiday-calendar difference). Reindexing +
      forward-fill silently fixes the 3 missing days and drops the 1 extra
      one, so copper ends up on the exact same calendar as everything else.

Valuation ratios (unlike every other category, these start life quarterly)
    1. Drop {T}_Dividend_Yield_-_Common_Stock_-_Net_-_Issue_Specific_-_%,_TTM
       (too many missing values -- see count_ms.py's report).
    2. Forward-fill the quarterly series itself (past values only), so any
       interior gap between reports is carried forward. Stays quarterly.
    3. Reindex onto the canonical calendar (union first, then narrow),
       forward-filling each ratio until the next report.
    (forward-fill happens on the FULL history BEFORE any date truncation --
    truncating first would throw away a report made before 2010-01-04 that
    is still needed to seed the earliest 2010 values, creating false
    missing values)

    Reads : val_ratios/{T}_val_ratios.csv                                    (quarterly)
    Writes: val.h/{T}_val_quarterly_ms.csv   quarterly, Dividend Yield dropped, ffilled
            val.h/{T}_val_daily_ms.csv       daily, canonical calendar

AAPL news sentiment (starts life as one row per news ARTICLE, not per day)
    1. Average same-day scores down to one score per calendar date (297 of
       the 1081 rows share a date with another article).
    2. Reindex onto the canonical calendar the same way as everything else.

    Reads : data_preprocess/feature_engineering/AAPL_news/AAPL_scores.csv
    Writes: AAPL_news_handle/AAPL_scores_h.csv   daily, canonical calendar

Prices / ETFs / macro / EWT / log returns / technical indicators
    Already daily; just reindexed onto the canonical calendar as described
    above (a no-op truncation for everything except copper).

    Reads : data_collection/data/collected/tickers/prices_v/{T}_prices_v.csv
            data_collection/data/collected/ETFs/{ETF}_prices_v.csv
            data_collection/data/collected/MACRO_IND/{name}_prices_v.csv
            data_preprocess/feature_engineering/EWT/{T}_ewt.csv
            data_preprocess/feature_engineering/tickers_log_ret/{T}_log_ret.csv
            data_preprocess/feature_engineering/technical_ind/{T}_technical_ind.csv
    Writes: ticker_prices_h/{T}_prices_h.csv
            ETFs_h/{ETF}_prices_h.csv
            macro_h/{name}_prices_h.csv
            EWT_h/{T}_ewt_h.csv
            log_returns_h/{T}_log_ret_h.csv
            technical_ind_h/{T}_technical_ind_h.csv

Run:
    python handling_ms.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

MASTERS = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

VAL_RATIOS_DIR = MASTERS / "data_preprocess/feature_engineering/val_ratios"
PRICES_DIR = MASTERS / "data_collection/data/collected/tickers/prices_v"
ETF_DIR = MASTERS / "data_collection/data/collected/ETFs"
SPY_DIR = MASTERS / "data_collection/data/collected/SPY"
MACRO_DIR = MASTERS / "data_collection/data/collected/MACRO_IND"
EWT_DIR = MASTERS / "data_preprocess/feature_engineering/EWT"
LOG_RET_DIR = MASTERS / "data_preprocess/feature_engineering/tickers_log_ret"
TECH_IND_DIR = MASTERS / "data_preprocess/feature_engineering/technical_ind"
AAPL_SCORES_PATH = MASTERS / "data_preprocess/feature_engineering/AAPL_news/AAPL_scores.csv"

VAL_OUT_DIR = HERE / "val.h"
NEWS_OUT_DIR = HERE / "AAPL_news_handle"
PRICES_OUT_DIR = HERE / "ticker_prices_h"
ETF_OUT_DIR = HERE / "ETFs_h"
SPY_OUT_DIR = HERE / "SPY_h"
MACRO_OUT_DIR = HERE / "macro_h"
EWT_OUT_DIR = HERE / "EWT_h"
LOG_RET_OUT_DIR = HERE / "log_returns_h"
TECH_IND_OUT_DIR = HERE / "technical_ind_h"

START = "2010-01-04"
END = "2024-12-31"

DIVIDEND_SUFFIX = "Dividend_Yield_-_Common_Stock_-_Net_-_Issue_Specific_-_%,_TTM"


def trading_days_for(ticker: str) -> pd.DatetimeIndex:
    prices = pd.read_csv(PRICES_DIR / f"{ticker}_prices_v.csv")
    return pd.DatetimeIndex(sorted(pd.to_datetime(prices[prices.columns[0]]).unique()))


def canonical_trading_days() -> pd.DatetimeIndex:
    """The one calendar every output is aligned to: AAPL's own trading
    days (identical across all 6 tickers and all 6 sector ETFs), cut down
    to the common window."""
    days = trading_days_for("AAPL")
    return days[(days >= pd.Timestamp(START)) & (days <= pd.Timestamp(END))]


def align_to_calendar(df: pd.DataFrame, canonical: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex onto the union of the canonical calendar and the frame's own
    dates, forward-fill across that union, then narrow to the calendar."""
    return df.reindex(canonical.union(df.index)).ffill().reindex(canonical)


# --------------------------------------------------------------------------- #
# Valuation ratios (quarterly -> daily)
# --------------------------------------------------------------------------- #

def process_valuation(ticker: str, out_dir: Path, canonical: pd.DatetimeIndex) -> None:
    val_path = VAL_RATIOS_DIR / f"{ticker}_val_ratios.csv"
    if not val_path.exists():
        print(f"[{ticker}] valuations: not found: {val_path}")
        return

    df = pd.read_csv(val_path, index_col=0, parse_dates=True).sort_index()
    df.index.name = "Date"

    div_col = f"{ticker}_{DIVIDEND_SUFFIX}"
    df = df.drop(columns=[c for c in [div_col] if c in df.columns])

    before = int(df.isna().sum().sum())
    df = df.ffill()
    after = int(df.isna().sum().sum())

    q_dst = out_dir / f"{ticker}_val_quarterly_ms.csv"
    df.to_csv(q_dst)
    print(f"[{ticker}] val quarterly: {len(df):>3} rows, missing {before} -> {after} "
          f"-> {q_dst}")

    daily = align_to_calendar(df, canonical)
    daily.index.name = "Date"

    d_dst = out_dir / f"{ticker}_val_daily_ms.csv"
    daily.to_csv(d_dst)
    n_missing = int(daily.isna().sum().sum())
    print(f"[{ticker}] val daily    : {len(daily)} rows "
          f"[{daily.index.min().date()} .. {daily.index.max().date()}], "
          f"missing {n_missing} -> {d_dst}")


# --------------------------------------------------------------------------- #
# AAPL news sentiment (per-article -> daily)
# --------------------------------------------------------------------------- #

def process_aapl_news(out_dir: Path, canonical: pd.DatetimeIndex) -> None:
    if not AAPL_SCORES_PATH.exists():
        print(f"[AAPL news] not found: {AAPL_SCORES_PATH}")
        return

    raw = pd.read_csv(AAPL_SCORES_PATH, parse_dates=["Date"])
    n_dup = len(raw) - raw["Date"].nunique()

    daily_scores = raw.groupby("Date")["AAPL_sentiment_score"].mean().to_frame()
    daily = align_to_calendar(daily_scores, canonical)
    daily.index.name = "Date"

    dst = out_dir / "AAPL_scores_h.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(dst)

    n_missing = int(daily.isna().sum().sum())
    print(f"[AAPL news] {len(raw)} articles ({n_dup} same-day duplicates averaged) "
          f"-> {len(daily)} trading days "
          f"[{daily.index.min().date()} .. {daily.index.max().date()}], "
          f"missing {n_missing} -> {dst}")


# --------------------------------------------------------------------------- #
# Already-daily categories: prices, ETFs, macro, EWT, log returns, technicals
# --------------------------------------------------------------------------- #

def align_file(src_path: Path, dst_path: Path, canonical: pd.DatetimeIndex,
              label: str) -> None:
    df = pd.read_csv(src_path, index_col=0, parse_dates=True).sort_index()
    df.index.name = "Date"
    before_rows = len(df)

    aligned = align_to_calendar(df, canonical)
    aligned.index.name = "Date"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(dst_path)

    n_missing = int(aligned.isna().sum().sum())
    print(f"[{label}] {before_rows} -> {len(aligned)} rows "
          f"[{aligned.index.min().date()} .. {aligned.index.max().date()}], "
          f"missing {n_missing} -> {dst_path}")


def align_folder(src_dir: Path, pattern: str, out_dir: Path,
                 canonical: pd.DatetimeIndex, rename: callable) -> None:
    for src_path in sorted(src_dir.glob(pattern)):
        dst_path = out_dir / rename(src_path.name)
        align_file(src_path, dst_path, canonical, label=src_path.stem)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Align every source onto one canonical trading-day calendar.")
    args = ap.parse_args()

    canonical = canonical_trading_days()
    print(f"canonical calendar: {len(canonical)} trading days "
          f"[{canonical.min().date()} .. {canonical.max().date()}]\n")

    VAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = sorted(p.name.removesuffix("_val_ratios.csv")
                      for p in VAL_RATIOS_DIR.glob("*_val_ratios.csv"))
    for t in tickers:
        process_valuation(t, VAL_OUT_DIR, canonical)

    NEWS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    process_aapl_news(NEWS_OUT_DIR, canonical)

    PRICES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(PRICES_DIR, "*_prices_v.csv", PRICES_OUT_DIR, canonical,
                rename=lambda n: n.replace("_prices_v.csv", "_prices_h.csv"))

    SPY_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(SPY_DIR, "*_prices_v.csv", SPY_OUT_DIR, canonical,
                rename=lambda n: n.replace("_prices_v.csv", "_prices_h.csv"))

    ETF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(ETF_DIR, "*_prices_v.csv", ETF_OUT_DIR, canonical,
                rename=lambda n: n.replace("_prices_v.csv", "_prices_h.csv"))

    MACRO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(MACRO_DIR, "*_prices_v.csv", MACRO_OUT_DIR, canonical,
                rename=lambda n: n.replace("_prices_v.csv", "_prices_h.csv"))

    EWT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(EWT_DIR, "*_ewt.csv", EWT_OUT_DIR, canonical,
                rename=lambda n: n.replace("_ewt.csv", "_ewt_h.csv"))

    LOG_RET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(LOG_RET_DIR, "*_log_ret.csv", LOG_RET_OUT_DIR, canonical,
                rename=lambda n: n.replace("_log_ret.csv", "_log_ret_h.csv"))

    TECH_IND_OUT_DIR.mkdir(parents=True, exist_ok=True)
    align_folder(TECH_IND_DIR, "*_technical_ind.csv", TECH_IND_OUT_DIR, canonical,
                rename=lambda n: n.replace("_technical_ind.csv", "_technical_ind_h.csv"))

    print("\ndone")


if __name__ == "__main__":
    main()
