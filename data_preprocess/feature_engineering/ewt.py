"""
Build leak-free EWT (Empirical Wavelet Transform) band features for each
ticker, stitched into a single continuous file -- no separate period1/
period2 outputs, no period-identifying column.

Method (three-step causal construction)
  Two rolling walk-forward windows are used to freeze boundaries without
  ever looking at a future day:
    period1: train 2010-01-02..2022-12-31, test 2023-01-01..2023-12-31
    period2: train 2011-01-02..2023-12-31, test 2024-01-01..2024-12-31
  For each window:
    Step 1  Freeze the K-1 EWT band boundaries from the TRAINING returns of
            that window only (boundaries never see a test-period day).
    Step 2  Sweep every day of the window; each day's K band values come
            from a trailing block ending on that day, split by the frozen
            boundaries (no future day enters any value).

  Kept separately, period1 and period2 overlap in time (2011-2022 falls in
  both), so the same calendar date would carry two different band values.
  Instead the two windows are STITCHED into one continuous, non-overlapping
  series per ticker: period1 supplies every date through its own test end
  (2023-12-29), and period2 supplies only the dates strictly after that --
  its 2024 test days, the one span it uniquely contributes. Every date
  therefore appears exactly once, so no period column is needed in the main
  output.

  The date bounds actually used for each window (train/test start and end,
  plus the frozen boundary values) go to a separate file, so that
  information is not lost by dropping the period column from the main data.

Reads : tickers_log_ret/{TICKER}_log_ret.csv   (col {TICKER}_LOGR_Adj_Close)
Writes: EWT/{TICKER}_ewt.csv        Date + {T}_EWT_band1..5, one row per
                                     trading day, no gaps or duplicate dates
        EWT/ewt_bounds.csv          per ticker/period: train/test start and
                                     end, and the frozen boundary values
        EWT/verify_no_leakage.csv   per ticker/period, two independent
                                     no-look-ahead checks (see below)

No-look-ahead is verified two independent ways per period:
  1. Perturbation test: shock every day after a chosen test day and confirm
     that day's band values do not move (proves causality directly).
  2. Predictive test: regress the NEXT day's return on the day's band
     values, out-of-sample; a leak-free feature scores OOS R^2 ~ 0.

Requires: numpy, pandas, ewtpy   (pip install ewtpy)

Run:
    python ewt.py
"""

from __future__ import annotations
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COLLECTED_ROOT = REPO_ROOT / "data_collection" / "data" / "collected"

import numpy as np
import pandas as pd
import ewtpy

TICKERS = ["AAPL", "JNJ", "JPM", "NKE", "XOM", "BA"]

K_BANDS = 5        # fixed number of bands; NEVER let this vary across windows
BLOCK = 250        # trailing-block length for the causal sweep (Step 2)

# Rolling windows: name -> (train_start, train_end, test_start, test_end) inclusive.
# Training starts a year before the intended span so the EWT warm-up (BLOCK
# days) is absorbed by the earliest year and does not consume any test days.
PERIODS = {
    "period1": ("2010-01-02", "2022-12-31", "2023-01-01", "2023-12-31"),
    "period2": ("2011-01-02", "2023-12-31", "2024-01-01", "2024-12-31"),
}


# --------------------------------------------------------------------------- #
# Step 1 - freeze the boundaries on the training returns
# --------------------------------------------------------------------------- #

def freeze_boundaries(train_returns: np.ndarray, k: int = K_BANDS) -> np.ndarray:
    """Derive the k-1 EWT band boundaries from TRAINING returns only."""
    train_returns = np.asarray(train_returns, dtype=float)
    train_returns = train_returns[~np.isnan(train_returns)]
    if len(train_returns) < BLOCK:
        raise ValueError(
            f"Training series too short ({len(train_returns)}) for BLOCK={BLOCK}."
        )
    _, _, boundaries = ewtpy.EWT1D(train_returns, N=k)
    return np.sort(np.asarray(boundaries, dtype=float))


# --------------------------------------------------------------------------- #
# Filter bank from frozen boundaries (used by the causal sweep)
# --------------------------------------------------------------------------- #

def _meyer_beta(x: np.ndarray) -> np.ndarray:
    b = np.zeros_like(x)
    mid = (x > 0) & (x < 1)
    xm = x[mid]
    b[mid] = xm ** 4 * (35 - 84 * xm + 70 * xm ** 2 - 20 * xm ** 3)
    b[x >= 1] = 1.0
    return b


def _transition_width(boundaries: np.ndarray) -> float:
    b = np.concatenate([[0.0], np.asarray(boundaries), [np.pi]])
    g = float(np.min(np.abs(np.diff(b)) / (b[1:] + b[:-1] + 1e-12)))
    return max(min(g, 0.5), 1e-3)


def _scaling(w: np.ndarray, w1: float, gamma: float) -> np.ndarray:
    t = gamma * w1
    f = np.zeros_like(w)
    f[w <= w1 - t] = 1.0
    band = (w >= w1 - t) & (w <= w1 + t)
    f[band] = np.cos(np.pi / 2 * _meyer_beta((w[band] - (w1 - t)) / (2 * t)))
    return f


def _wavelet(w: np.ndarray, wl: float, wh: float, gamma: float) -> np.ndarray:
    tl, th = gamma * wl, gamma * wh
    f = np.zeros_like(w)
    passband = (w >= wl + tl) & (w <= wh - th)
    f[passband] = 1.0
    up = (w >= wl - tl) & (w <= wl + tl)
    f[up] = np.sin(np.pi / 2 * _meyer_beta((w[up] - (wl - tl)) / (2 * tl)))
    dn = (w >= wh - th) & (w <= wh + th)
    f[dn] = np.cos(np.pi / 2 * _meyer_beta((w[dn] - (wh - th)) / (2 * th)))
    return f


def _ewt_filter_bank(boundaries: np.ndarray, n: int) -> np.ndarray:
    """Empirical Meyer scaling + wavelet filters on an FFT grid of length n."""
    w = np.abs(np.fft.fftfreq(n) * 2 * np.pi)
    bnd = list(boundaries) + [np.pi]
    k = len(bnd)
    gamma = _transition_width(boundaries)
    filt = np.zeros((k, n))

    filt[0] = _scaling(w, bnd[0], gamma)
    for m in range(1, k):
        filt[m] = _wavelet(w, bnd[m - 1], bnd[m], gamma)

    power = np.sqrt((filt ** 2).sum(axis=0))
    power[power < 1e-12] = 1.0
    return filt / power


# --------------------------------------------------------------------------- #
# Step 2 - causal sweep: one row of K band values per day
# --------------------------------------------------------------------------- #

def causal_band_values(returns: np.ndarray, boundaries: np.ndarray,
                       block: int = BLOCK, k: int = K_BANDS) -> np.ndarray:
    """
    For each day t with >= `block` prior samples, compute the K band values
    at t from returns[t-block+1 : t+1] using the FROZEN boundaries. Only the
    last position of the reconstructed block is kept, so no future sample
    can enter. Returns (n, k); the first `block-1` rows are NaN (warm-up).
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    out = np.full((n, k), np.nan)
    bank = _ewt_filter_bank(boundaries, block)

    for t in range(block - 1, n):
        seg = returns[t - block + 1: t + 1]
        if np.isnan(seg).any():
            continue
        F = np.fft.fft(seg)
        for m in range(k):
            band = np.real(np.fft.ifft(F * bank[m]))
            out[t, m] = band[-1]
    return out


# --------------------------------------------------------------------------- #
# Loading, per-period construction, stitching
# --------------------------------------------------------------------------- #

def load_log_returns(path: Path, ticker: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    col = f"{ticker}_LOGR_Adj_Close"
    if col not in df.columns:
        raise KeyError(f"{col} not in {path}; have {list(df.columns)}")
    return df[col].dropna()


def build_period(returns: pd.Series, train_start: str, train_end: str,
                 test_start: str, test_end: str,
                 block: int = BLOCK, k: int = K_BANDS) -> pd.DataFrame:
    """
    Boundaries are frozen on [train_start, train_end]. Band values are
    produced for every day in [train_start, test_end]; the causal sweep is
    warmed up on the BLOCK days that PRECEDE train_start (available since
    the log-return series begins earlier), so train_start itself already
    has a valid value.
    """
    train = returns.loc[train_start:train_end]
    if len(train) < block:
        raise ValueError(f"Training span has {len(train)} rows; need >= block={block}.")

    boundaries = freeze_boundaries(train.values, k=k)

    ts_pos = returns.index.searchsorted(pd.Timestamp(train_start))
    warm_start = max(0, ts_pos - (block - 1))
    swept = returns.iloc[warm_start:].loc[:test_end]
    vals = causal_band_values(swept.values, boundaries, block=block, k=k)

    cols = [f"EWT_band{m + 1}" for m in range(k)]
    out = pd.DataFrame(vals, index=swept.index, columns=cols)
    out = out.loc[train_start:test_end].dropna()
    out.attrs["boundaries"] = boundaries
    return out


def stitch_periods(period_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine the rolling windows into one continuous, non-overlapping series:
    the first window supplies every date through its own last date; each
    later window supplies only the dates strictly after that -- its unique
    contribution -- so no date appears twice and no period label is needed.
    """
    names = list(period_frames)
    combined = period_frames[names[0]].copy()
    combined.attrs = {}
    for name in names[1:]:
        nxt = period_frames[name].copy()
        nxt.attrs = {}
        combined = pd.concat([combined, nxt[nxt.index > combined.index.max()]])
    return combined.sort_index()


# --------------------------------------------------------------------------- #
# Leakage checks (run per period, before stitching)
# --------------------------------------------------------------------------- #

def perturbation_check(returns: pd.Series, train_start: str, train_end: str,
                       test_start: str, test_end: str) -> dict:
    """
    Freeze boundaries on the window's training span, compute band values
    over the full available series from train_start onward, then perturb
    every day AFTER a test day and confirm that day's values do not move.
    """
    train = returns.loc[train_start:train_end]
    bnd = freeze_boundaries(train.values)

    full = returns.loc[train_start:]
    base = causal_band_values(full.values, bnd)

    test_mask = (full.index >= pd.Timestamp(test_start)) & (full.index <= pd.Timestamp(test_end))
    test_pos = np.where(test_mask)[0]
    results = {}

    for label, pos in [("last_test_day", int(test_pos[-1])),
                       ("mid_test_day", int(test_pos[len(test_pos) // 2]))]:
        if pos + 1 >= len(full):
            results[label] = ("no future day in series", 0.0)
            continue
        pert = full.values.copy()
        pert[pos + 1:] += 5.0
        p2 = causal_band_values(pert, bnd)
        delta = float(np.nan_to_num(np.abs(p2[pos] - base[pos])).max())
        n_future = len(full) - 1 - pos
        results[label] = (f"{full.index[pos].date()} ({n_future} future days)", delta)
    return results


def predictive_check(period_frame: pd.DataFrame, returns: pd.Series) -> float:
    """Regress the NEXT day's return on the period's own band values, OOS."""
    band_cols = [c for c in period_frame.columns if c.startswith("EWT_band")]
    X = period_frame[band_cols].values
    y = returns.reindex(period_frame.index).shift(-1).values
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[ok], y[ok]
    if len(X) < 50:
        return float("nan")

    Xd = np.c_[X, np.ones(len(X))]
    cut = int(0.7 * len(Xd))
    beta, *_ = np.linalg.lstsq(Xd[:cut], y[:cut], rcond=None)
    pred = Xd[cut:] @ beta
    yt = y[cut:]
    return 1 - np.sum((yt - pred) ** 2) / np.sum((yt - yt.mean()) ** 2)


# --------------------------------------------------------------------------- #
# Per-ticker orchestration
# --------------------------------------------------------------------------- #

def build_ticker(ticker: str, log_root: Path, out_root: Path,
                 block: int, k: int):
    src = log_root / f"{ticker}_log_ret.csv"
    if not src.exists():
        print(f"[{ticker}] log-return file not found: {src}")
        return None

    returns = load_log_returns(src, ticker)

    period_frames: dict[str, pd.DataFrame] = {}
    bounds_rows: list[dict] = []

    for pname, (tr_s, tr_e, te_s, te_e) in PERIODS.items():
        try:
            pf = build_period(returns, tr_s, tr_e, te_s, te_e, block, k)
        except (ValueError, KeyError) as e:
            print(f"[{ticker}] {pname}: skipped ({e})")
            continue
        period_frames[pname] = pf
        bounds_rows.append({
            "ticker": ticker, "period": pname,
            "train_start": tr_s, "train_end": tr_e,
            "test_start": te_s, "test_end": te_e,
            "data_start": pf.index.min().date().isoformat(),
            "data_end": pf.index.max().date().isoformat(),
            "boundaries": np.round(pf.attrs["boundaries"], 4).tolist(),
        })

    if not period_frames:
        print(f"[{ticker}] no periods built.")
        return None

    stitched = stitch_periods(period_frames)
    band_cols = [c for c in stitched.columns if c.startswith("EWT_band")]
    out = stitched[band_cols].copy()
    out.columns = [f"{ticker}_{c}" for c in band_cols]
    out.index.name = "Date"

    dst = out_root / f"{ticker}_ewt.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst)
    print(f"[{ticker}] {len(out)} rows "
          f"[{out.index.min().date()} .. {out.index.max().date()}] -> {dst}")

    return out, bounds_rows, period_frames, returns


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Build leak-free, stitched EWT band features.")
    ap.add_argument("--log-root", default=str(HERE / "tickers_log_ret"),
                    help="folder containing {T}_log_ret.csv")
    ap.add_argument("--out-root", default=str(HERE / "EWT"))
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--k", type=int, default=K_BANDS)
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    log_root = Path(args.log_root)
    out_root = Path(args.out_root)

    all_bounds: list[dict] = []
    all_leak: list[dict] = []

    for t in args.tickers:
        result = build_ticker(t, log_root, out_root, args.block, args.k)
        if result is None:
            continue
        _out, bounds_rows, period_frames, returns = result
        all_bounds.extend(bounds_rows)

        for pname, pf in period_frames.items():
            tr_s, tr_e, te_s, te_e = PERIODS[pname]

            pchk = perturbation_check(returns, tr_s, tr_e, te_s, te_e)
            for label, (day, delta) in pchk.items():
                verdict = "CLEAN" if delta <= 1e-9 else "LEAK"
                all_leak.append({"ticker": t, "period": pname,
                                "check": f"perturbation_{label}", "detail": day,
                                "value": delta, "verdict": verdict})

            r2 = predictive_check(pf, returns)
            verdict = "CLEAN" if (np.isnan(r2) or r2 < 0.02) else "SUSPICIOUS"
            all_leak.append({"ticker": t, "period": pname,
                            "check": "predictive_oos_r2", "detail": "",
                            "value": r2, "verdict": verdict})

    out_root.mkdir(parents=True, exist_ok=True)

    bounds_df = pd.DataFrame(all_bounds)
    bounds_path = out_root / "ewt_bounds.csv"
    bounds_df.to_csv(bounds_path, index=False)
    print(f"\nbounds -> {bounds_path}")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(bounds_df.to_string(index=False))

    leak_df = pd.DataFrame(all_leak)
    leak_path = out_root / "verify_no_leakage.csv"
    leak_df.to_csv(leak_path, index=False)
    print(f"\nleakage verification -> {leak_path}")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.max_rows", None):
        print(leak_df.to_string(index=False))

    all_clean = bool((leak_df["verdict"] == "CLEAN").all()) if not leak_df.empty else False
    print("\n" + "=" * 50)
    print("OVERALL:", "NO LEAKAGE DETECTED" if all_clean
          else "POSSIBLE LEAKAGE - INVESTIGATE")


if __name__ == "__main__":
    main()
