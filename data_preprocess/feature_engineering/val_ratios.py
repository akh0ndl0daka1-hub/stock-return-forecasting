"""
Convert the raw valuation-ratio Excel workbooks into per-ticker CSVs.

Reads : <Masters>/data_collection/data/collected/tickers/Raw_valuation_ratios/{TICKER}_Valuation.xlsx
        (sheet "Sheet1")
Writes: val_ratios/{TICKER}_val_ratios.csv

Transformations
    1. Date written as YYYY-MM-DD.
    2. Column names ticker-prefixed and spaces replaced with underscores,
       e.g. "Free Cash Flow Yield - %, TTM" ->
            AAPL_Free_Cash_Flow_Yield_-_%,_TTM
    3. Accounting-style negatives unwrapped from parentheses:
       "(2.94)"  -> "-2.94"   (parens mean negative)
       "(-2.94)" -> "-2.94"   (already signed; just drop the brackets)

Run:
    python val_ratios.py
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTED_ROOT = REPO_ROOT / "data_collection" / "data" / "collected"

import pandas as pd

SRC_DEFAULT = COLLECTED_ROOT / "tickers" / "Raw_valuation_ratios"
OUT_DEFAULT = HERE / "val_ratios"

PAREN_RE = re.compile(r"^\((-?[0-9.]+)\)$")


def unwrap_parens(x):
    if isinstance(x, str):
        m = PAREN_RE.match(x.strip())
        if m:
            inner = m.group(1)
            return inner if inner.startswith("-") else "-" + inner
    return x


def collect(ticker: str, src_dir: Path, out_dir: Path) -> None:
    src = src_dir / f"{ticker}_Valuation.xlsx"
    if not src.exists():
        print(f"  ! not found: {src}")
        return

    df = pd.read_excel(src, sheet_name="Sheet1")
    date_col = df.columns[0]
    value_cols = df.columns[1:]

    # 3. accounting-style negatives -> numeric
    df[value_cols] = df[value_cols].map(unwrap_parens)
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1. Date -> YYYY-MM-DD
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(date_col)

    # 2. ticker-prefixed columns, spaces -> underscores
    df.columns = [date_col] + [f"{ticker}_{c.replace(' ', '_')}" for c in value_cols]

    dst = out_dir / f"{ticker}_val_ratios.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)

    n_missing = int(df.isna().sum().sum())
    print(f"  {ticker}: rows={len(df)} cols={df.shape[1]} missing={n_missing} "
          f"start={df[date_col].min()} end={df[date_col].max()} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect valuation-ratio xlsx into CSVs.")
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)

    tickers = sorted(p.name.removesuffix("_Valuation.xlsx")
                      for p in src_dir.glob("*_Valuation.xlsx"))
    for t in tickers:
        collect(t, src_dir, out_dir)
    print("\ndone")


if __name__ == "__main__":
    main()
