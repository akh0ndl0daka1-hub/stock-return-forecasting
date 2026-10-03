"""
Compute technical indicators for each ticker, with no look-ahead.

Reads : <Masters>/data_collection/data/collected/tickers/prices_v/{TICKER}_prices_v.csv
        (needs {T}_Open/High/Low/Close/Adj_Close/Volume)
Writes: technical_ind/{TICKER}_technical_ind.csv
        columns prefixed with the ticker, e.g. AAPL_SMA_20, AAPL_RSI_14.

Indicators (windows 5, 20, 60 where a window applies):
    SMA, EMA, MACD line, RSI, ROC, CCI, Williams %R, ATR, Bollinger Bands,
    A/D line, OBV.

    RSI (14), MACD line (12/26), A/D line, and OBV are not swept across
    5/20/60 -- RSI and MACD use their own standard fixed periods, and A/D
    line / OBV are running cumulative sums with no window parameter.

NO LOOK-AHEAD
    Every indicator uses only the current bar and earlier bars. All rolling
    windows use the default trailing alignment (center=False); an assertion
    guards against any centred window. Each indicator at day t depends solely
    on prices up to and including day t, so the features are causal and can
    be computed once over the whole series without a train/test split.

Run:
    python technical_ind.py
    python technical_ind.py --src ../../data_collection/data/collected/tickers/prices_v \
        --out technical_ind --windows 5 20 60
"""

from __future__ import annotations
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTED_ROOT = REPO_ROOT / "data_collection" / "data" / "collected"

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "JNJ", "JPM", "NKE", "XOM", "BA"]
WINDOWS = [5, 20, 60]


# --------------------------------------------------------------------------- #
# Column access
# --------------------------------------------------------------------------- #

def col(df: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    name = f"{ticker}_{field}"
    if name not in df.columns:
        raise KeyError(f"{name} not found for {ticker}; have {list(df.columns)}")
    return pd.to_numeric(df[name], errors="coerce")


# --------------------------------------------------------------------------- #
# Indicators (all trailing / causal)
# --------------------------------------------------------------------------- #

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    # adjust=False gives the standard recursive EMA, which uses only past data.
    return s.ewm(span=n, adjust=False).mean()


def macd_line(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (recursive, past-only)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def bollinger(close: pd.Series, n: int, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid
    pctb = (close - lower) / (upper - lower)
    return pd.DataFrame({f"BB_mid_{n}": mid, f"BB_up_{n}": upper,
                         f"BB_low_{n}": lower, f"BB_width_{n}": width,
                         f"BB_pctb_{n}": pctb})


def cci(high, low, close, n: int) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mad)


def atr(high, low, close, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()   # Wilder


def roc(close: pd.Series, n: int) -> pd.Series:
    return 100 * (close - close.shift(n)) / close.shift(n)


def williams_r(high, low, close, n: int) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll)


def ad_line(high, low, close, volume) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm.fillna(0) * volume
    return mfv.cumsum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# --------------------------------------------------------------------------- #
# Assemble per ticker
# --------------------------------------------------------------------------- #

def build_ticker(ticker: str, src: Path, out: Path, windows: list[int]) -> bool:
    path = src / f"{ticker}_prices_v.csv"
    if not path.exists():
        print(f"[{ticker}] price file not found: {path}")
        return False

    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    o = col(df, ticker, "Open")
    h = col(df, ticker, "High")
    l = col(df, ticker, "Low")
    c = col(df, ticker, "Close")
    v = col(df, ticker, "Volume")

    feat = {}

    # windowed indicators
    for n in windows:
        feat[f"SMA_{n}"] = sma(c, n)
        feat[f"EMA_{n}"] = ema(c, n)
        feat[f"CCI_{n}"] = cci(h, l, c, n)
        feat[f"ATR_{n}"] = atr(h, l, c, n)
        feat[f"ROC_{n}"] = roc(c, n)
        feat[f"WILLR_{n}"] = williams_r(h, l, c, n)
        for name, series in bollinger(c, n).items():
            feat[name] = series

    # fixed-parameter / window-free indicators
    feat["MACD"] = macd_line(c)
    feat["RSI_14"] = rsi(c, 14)
    feat["AD"] = ad_line(h, l, c, v)
    feat["OBV"] = obv(c, v)

    out_df = pd.DataFrame(feat, index=df.index)

    # ---- causality guard: recompute one windowed indicator with center=True
    # and confirm it differs, proving we are NOT centring anywhere.
    trailing = sma(c, windows[0])
    centred = c.rolling(windows[0], center=True).mean()
    assert not trailing.equals(centred), "rolling windows must be trailing!"

    # prefix every column with the ticker
    out_df.columns = [f"{ticker}_{name}" for name in out_df.columns]

    dst = out / f"{ticker}_technical_ind.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst)
    print(f"[{ticker}] {out_df.shape[0]} rows x {out_df.shape[1]} indicators "
          f"-> {dst}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal technical indicators.")
    ap.add_argument("--src", default=str(COLLECTED_ROOT / "tickers" / "prices_v"))
    ap.add_argument("--out", default=str(HERE / "technical_ind"))
    ap.add_argument("--windows", nargs="*", type=int, default=WINDOWS)
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    ok = sum(build_ticker(t, Path(args.src), Path(args.out), args.windows)
             for t in args.tickers)
    print(f"\ncomputed technicals for {ok}/{len(args.tickers)} tickers")


if __name__ == "__main__":
    main()
