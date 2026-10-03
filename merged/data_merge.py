"""
Merge every aligned (_h) source into one file per ticker.

All inputs come from Masters/Missing_values/*_h, which were already
reindexed onto the same canonical trading-day calendar (2010-01-04 ..
2024-12-31, 3774 rows -- see handling_ms.py), so this is a plain
column-wise concat on a shared Date index, not a date-matching join.

Per ticker, merges:
    ticker prices        Missing_values/ticker_prices_h/{T}_prices_h.csv
    valuation ratios      Missing_values/val.h/{T}_val_daily_ms.csv         (that company's own)
    sector ETF             Missing_values/ETFs_h/{ETF}_prices_h.csv         (that company's own sector only,
                                                                              e.g. AAPL -> Technology/XLK)
    SPY                    Missing_values/SPY_h/SPY_prices_h.csv            (every ticker)
    macro indicators (x7)  Missing_values/macro_h/{name}_prices_h.csv       (every ticker)
    technical indicators   Missing_values/technical_ind_h/{T}_technical_ind_h.csv  (that company's own)
    log return              Missing_values/log_returns_h/{T}_log_ret_h.csv  (that company's own)
    EWT bands               Missing_values/EWT_h/{T}_ewt_h.csv              (that company's own)

AAPL additionally merges its news sentiment score
    (Missing_values/AAPL_news_handle/AAPL_scores_h.csv), so AAPL's merged
    file has one extra column the other five tickers don't have.

Writes: {T}_merged.csv (this same folder)

Run:
    python data_merge.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

MASTERS = Path(__file__).resolve().parent.parent
MS = MASTERS / "Missing_values"

TICKERS = ["AAPL", "JNJ", "JPM", "NKE", "XOM", "BA"]
SECTOR_ETF = {"AAPL": "XLK", "JNJ": "XLV", "JPM": "XLF",
             "NKE": "XLY", "XOM": "XLE", "BA": "XLI"}
MACRO = ["vix", "us_dollar_index", "copper", "crude_oil", "gold",
         "treasury_20y", "treasury_1_3y"]


def read_dated(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ! not found: {path}")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def merge_ticker(ticker: str, out_dir: Path) -> None:
    frames: list[pd.DataFrame] = []

    frames.append(read_dated(MS / "ticker_prices_h" / f"{ticker}_prices_h.csv"))
    frames.append(read_dated(MS / "val.h" / f"{ticker}_val_daily_ms.csv"))

    etf = SECTOR_ETF[ticker]
    frames.append(read_dated(MS / "ETFs_h" / f"{etf}_prices_h.csv"))

    frames.append(read_dated(MS / "SPY_h" / "SPY_prices_h.csv"))

    for m in MACRO:
        frames.append(read_dated(MS / "macro_h" / f"{m}_prices_h.csv"))

    frames.append(read_dated(MS / "technical_ind_h" / f"{ticker}_technical_ind_h.csv"))
    frames.append(read_dated(MS / "log_returns_h" / f"{ticker}_log_ret_h.csv"))
    frames.append(read_dated(MS / "EWT_h" / f"{ticker}_ewt_h.csv"))

    if ticker == "AAPL":
        frames.append(read_dated(MS / "AAPL_news_handle" / "AAPL_scores_h.csv"))

    if any(f is None for f in frames):
        print(f"[{ticker}] skipped: missing source(s)")
        return

    merged = pd.concat(frames, axis=1, join="inner").sort_index()

    n_missing = int(merged.isna().sum().sum())
    dst = out_dir / f"{ticker}_merged.csv"
    merged.to_csv(dst)
    print(f"[{ticker}] {merged.shape[0]} rows x {merged.shape[1]} cols, "
          f"missing={n_missing}, "
          f"[{merged.index.min().date()} .. {merged.index.max().date()}] -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge all aligned sources per ticker.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in args.tickers:
        merge_ticker(t, out_dir)

    print("\ndone")


if __name__ == "__main__":
    main()
