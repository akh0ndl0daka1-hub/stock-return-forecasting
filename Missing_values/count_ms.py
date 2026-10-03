"""
Survey every data source in the Masters pipeline and report, per file:
data type, start date, finish date, total rows, total columns, rows
containing any missing value, and columns containing any missing value.

This is an ANALYSIS step only -- nothing is filled or dropped here. The
goal is to see, before doing any missing-value handling, which files
already share a common trading-day calendar and which don't, so all files
can eventually be aligned to the same rows for modeling.

Sources surveyed (glob *.csv within each folder, one row per file):
    prices             data_collection/data/collected/tickers/prices_v
    valuation ratios   data_preprocess/feature_engineering/val_ratios
    AAPL news score    data_preprocess/feature_engineering/AAPL_news/AAPL_scores.csv
    ETFs               data_collection/data/collected/ETFs
    SPY                data_collection/data/collected/SPY
    macro              data_collection/data/collected/MACRO_IND
    EWT                data_preprocess/feature_engineering/EWT
    log return         data_preprocess/feature_engineering/tickers_log_ret
    technical ind.     data_preprocess/feature_engineering/technical_ind

Note: EWT and log-return also each contain a companion report file
(ewt_bounds.csv / verify_no_leakage.csv, and stationary_evidence.csv) that
the *.csv glob picks up too. These are metadata, not date-indexed feature
series, so they show "-" for start/finish date rather than being excluded
silently -- that itself is a useful signal they don't belong with the
per-ticker feature files.

Writes: missing_values_report.csv (this same folder)

Run:
    python count_ms.py
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import pandas as pd

MASTERS = Path(__file__).resolve().parent.parent

SOURCES = {
    "prices":            MASTERS / "data_collection/data/collected/tickers/prices_v",
    "valuation_ratios":  MASTERS / "data_preprocess/feature_engineering/val_ratios",
    "ETF":               MASTERS / "data_collection/data/collected/ETFs",
    "SPY":               MASTERS / "data_collection/data/collected/SPY",
    "macro":             MASTERS / "data_collection/data/collected/MACRO_IND",
    "EWT":               MASTERS / "data_preprocess/feature_engineering/EWT",
    "log_return":        MASTERS / "data_preprocess/feature_engineering/tickers_log_ret",
    "technical_ind":     MASTERS / "data_preprocess/feature_engineering/technical_ind",
}

# single named file, not a folder glob
AAPL_NEWS_SCORES = MASTERS / "data_preprocess/feature_engineering/AAPL_news/AAPL_scores.csv"


def analyse(path: Path, data_type: str) -> dict:
    df = pd.read_csv(path)
    rows, cols = df.shape

    date_col = df.columns[0] if cols else None
    if date_col is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # non-date first columns (report files)
            dates = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dates = pd.Series([], dtype="datetime64[ns]")
    valid_dates = dates.dropna()
    start = valid_dates.min().date().isoformat() if len(valid_dates) else "-"
    finish = valid_dates.max().date().isoformat() if len(valid_dates) else "-"

    rows_with_missing = int(df.isna().any(axis=1).sum())
    cols_with_missing = int(df.isna().any(axis=0).sum())

    return {
        "data_type": data_type,
        "file": path.name,
        "start_date": start,
        "finish_date": finish,
        "total_rows": rows,
        "total_columns": cols,
        "rows_with_missing": rows_with_missing,
        "columns_with_missing": cols_with_missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Survey missing values across the Masters pipeline.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "missing_values_report.csv"))
    args = ap.parse_args()

    rows: list[dict] = []

    for data_type, folder in SOURCES.items():
        if not folder.exists():
            print(f"! folder not found: {folder}")
            continue
        for path in sorted(folder.glob("*.csv")):
            rows.append(analyse(path, data_type))

    if AAPL_NEWS_SCORES.exists():
        rows.append(analyse(AAPL_NEWS_SCORES, "AAPL_news_score"))
    else:
        print(f"! not found: {AAPL_NEWS_SCORES}")

    report = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)

    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.max_rows", None):
        print(report.to_string(index=False))

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
