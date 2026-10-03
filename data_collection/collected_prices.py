"""
Data collection for the study universe (2009-2024).

Collects, via Yahoo Finance:
  * 6 stocks                                - OHLCV + adjusted close
  * 6 sector ETFs (one per stock's sector)  - OHLCV + adjusted close
  * SPY                                     - OHLCV + adjusted close
  * 7 macro indicators                      - OHLCV + adjusted close where available

Saved layout (created under --root, default data_collection/data/collected):
  data/collected/tickers/prices_v/{ticker}_prices_v.csv
  data/collected/ETFs/{etf}_prices_v.csv
  data/collected/SPY/SPY_prices_v.csv
  data/collected/MACRO_IND/{name}_prices_v.csv

Every column is prefixed by its own ticker/ETF/SPY/macro name (uppercase
for macro), e.g. AAPL_Open, XLK_Adj_Close, SPY_Volume, VIX_Close.

Notes
  * auto_adjust=False so BOTH raw OHLC and a separate 'Adj Close' are saved.
  * Yahoo's `end` is EXCLUSIVE, so the day after --end is passed internally.
  * Gold, crude, and USD exposures use ETF proxies (GLD/USO/UUP); the 20-year
    and 1-3 year Treasury exposures use the TLT and SHY ETFs. Copper is the
    ONE exception: it uses the front-month futures HG=F, because the only
    copper ETF (CPER) launched Nov 2011 and cannot cover 2009-2024. VIX is
    also kept as the raw index (^VIX), not a futures-based ETF/ETN (e.g.
    VXX, VIXY): those roll VIX futures continuously and suffer persistent
    contango decay, tracking spot VIX poorly over a long sample, and several
    don't even cover the full 2009-2024 window.

Run:
    pip install yfinance pandas
    python collected_data.py
    python collected_data.py --root data/collected --start 2009-01-01 --end 2024-12-31
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "data" / "collected"


# --------------------------------------------------------------------------- #
# What to collect
# --------------------------------------------------------------------------- #

# stock -> its sector SPDR ETF
STOCKS = {
    "AAPL": "XLK",   # Technology
    "JNJ":  "XLV",   # Health Care
    "JPM":  "XLF",   # Financials
    "NKE":  "XLY",   # Consumer Discretionary
    "XOM":  "XLE",   # Energy
    "BA":   "XLI",   # Industrials
}

# macro indicator name -> Yahoo symbol
MACRO = {
    "vix":             "^VIX",      # CBOE volatility index (no adequate long-history ETF proxy)
    "treasury_20y":    "TLT",       # iShares 20+ Year Treasury Bond ETF
    "treasury_1_3y":   "SHY",       # iShares 1-3 Year Treasury Bond ETF
    "gold":            "GLD",       # SPDR Gold Shares
    "us_dollar_index": "UUP",       # Invesco DB US Dollar Index Bullish Fund
    "crude_oil":       "USO",       # United States Oil Fund
    "copper":          "HG=F",      # copper futures (no long-history copper ETF)
}

SPY = "SPY"


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #

def download(symbol: str, start: str, end: str, prefix: str,
             retries: int = 3) -> pd.DataFrame:
    """
    Download one symbol with retries. auto_adjust=False keeps raw OHLC and a
    separate 'Adj Close'. Yahoo's end is exclusive, so callers pass end+1 day.
    Returns a tidy frame indexed by Date with columns prefixed by `prefix` and
    spaces replaced by underscores, e.g. AAPL_Open, AAPL_Adj_Close, VIX_Close.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol, start=start, end=end,
                progress=False, auto_adjust=False, actions=False,
            )
            if df is not None and len(df):
                # yfinance may return a MultiIndex (field, ticker); flatten it.
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index.name = "Date"
                keep = [c for c in ["Open", "High", "Low", "Close",
                                    "Adj Close", "Volume"] if c in df.columns]
                df = df[keep].sort_index()
                # Prefix every column and turn spaces into underscores:
                # "Adj Close" -> "{prefix}_Adj_Close", "Open" -> "{prefix}_Open".
                df.columns = [f"{prefix}_{c.replace(' ', '_')}" for c in df.columns]
                return df
            last_err = "empty frame"
        except Exception as e:                       # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {symbol}: {last_err}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"  saved {len(df):>5} rows  ->  {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Collect market data for the study universe.")
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="output root folder")
    ap.add_argument("--start", default="2009-01-01")
    ap.add_argument("--end", default="2024-12-31",
                    help="inclusive last date")
    args = ap.parse_args()

    root = Path(args.root)
    start = args.start
    # Yahoo's `end` is exclusive; add a day so the inclusive last date is kept.
    end_excl = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    failures: list[str] = []

    def grab(symbol: str, dest: Path, label: str, prefix: str) -> None:
        print(f"[{label}] {symbol}  (cols: {prefix}_*)")
        try:
            df = download(symbol, start, end_excl, prefix=prefix)
            save_csv(df, dest)
        except Exception as e:                        # noqa: BLE001
            print(f"  !! {e}")
            failures.append(f"{label}:{symbol}")

    # 1) stocks -> data/collected/tickers/{ticker}_prices_v.csv
    for ticker in STOCKS:
        grab(ticker, root / "tickers" / "prices_v" /f"{ticker}_prices_v.csv",
             "stock", prefix=ticker)

    # 2) sector ETFs -> data/collected/ETFs/{etf}_prices_v.csv
    for etf in dict.fromkeys(STOCKS.values()):
        grab(etf, root / "ETFs" / f"{etf}_prices_v.csv",
             "etf", prefix=etf)

    # 3) SPY -> data/collected/SPY/SPY_prices_v.csv
    grab(SPY, root / "SPY" / "SPY_prices_v.csv", "spy", prefix=SPY)

    # 4) macro indicators -> data/collected/MACRO_IND/{name}_prices_v.csv,
    #    columns prefixed UPPERCASE (e.g. COPPER_Open, VIX_Close).
    for name, symbol in MACRO.items():
        grab(symbol, root / "MACRO_IND" / f"{name}_prices_v.csv",
             "macro", prefix=name.upper())

    # summary
    print("\n" + "=" * 60)
    total = len(STOCKS) + len(set(STOCKS.values())) + 1 + len(MACRO)
    print(f"collected {total - len(failures)}/{total} series")
    if failures:
        print("FAILED:", ", ".join(failures))
        print("Re-run to retry failed symbols; check the ticker still exists.")
    else:
        print("all series collected successfully.")


if __name__ == "__main__":
    main()
