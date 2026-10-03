"""
Prune redundant features via pairwise feature-feature correlation, per
ticker's merged dataset.

For each ticker, computes the Pearson correlation matrix across all
candidate feature columns -- excluding the log-return column, which is the
prediction target and is therefore never dropped and never used to justify
dropping another column (correlation with the target is signal, not
redundancy). Wherever two features correlate at |r| >= threshold (default
0.95; 0.90 is also common), the later of the two (in the dataset's own
column order) is dropped, since the pair carries near-identical
information and only one is needed. This collapses larger mutually
correlated clusters (e.g. SMA_5/SMA_20/SMA_60 all highly correlated with
one another) down to a single representative column, since every later
member of the cluster correlates with the first-kept one.

Adj_Close is likewise never dropped for any ticker, even if it would
otherwise be flagged as redundant with Open/High/Low/Close -- it is needed
downstream to reconstruct price levels from the predicted log return.

The correlation matrix is computed on the TRAINING window only -- dates up
to and including 2022-12-31, i.e. before period 1's test year (2023) and
period 2's test year (2024) -- never on either test window, so the choice
of which features to drop is not informed by test-period data. The
resulting drop list is then applied to the full dataset (train and both
test windows alike), so every split ends up with the same, consistently
reduced column set.

Reads : merged/{T}_merged.csv
Writes: {T}_reduced.csv               pruned feature set (target and
                                       Adj_Close both always kept)
        dropped_features_report.csv   ticker, dropped feature, the kept
                                       feature it correlated with, and r

Run:
    python feature_reduction.py
    python feature_reduction.py --threshold 0.9
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MASTERS = Path(__file__).resolve().parent.parent
MERGED_DIR = MASTERS / "merged"

TICKERS = ["AAPL", "JNJ", "JPM", "NKE", "XOM", "BA"]
THRESHOLD_DEFAULT = 0.95

# Last training date before either rolling period's test window (period 1
# tests 2023, period 2 tests 2024): the correlation matrix is fit on data
# up to and including this date only, to avoid leaking test-period
# information into the feature-selection decision.
TRAIN_END = "2022-12-31"


def target_column(ticker: str, columns: list[str]) -> str:
    col = f"{ticker}_LOGR_Adj_Close"
    if col not in columns:
        raise KeyError(f"target column {col} not found among {columns}")
    return col


def prune_ticker(ticker: str, out_dir: Path, threshold: float) -> list[dict]:
    src = MERGED_DIR / f"{ticker}_merged.csv"
    if not src.exists():
        print(f"[{ticker}] not found: {src}")
        return []

    df = pd.read_csv(src, index_col=0, parse_dates=True)
    target = target_column(ticker, list(df.columns))
    feature_cols = [c for c in df.columns if c != target]

    # Adj_Close is needed downstream to reconstruct price levels from the
    # predicted log return, so it participates in the correlation analysis
    # (it can still be the reference other columns are dropped against) but
    # is never itself dropped, regardless of its position in column order.
    # The sentiment score (AAPL only) is protected the same way, so it is
    # never silently pruned as "redundant" -- it needs to survive so a
    # with/without-news ablation can drop it manually and compare results.
    adj_close = f"{ticker}_Adj_Close"
    sentiment = f"{ticker}_sentiment_score"
    protected = {c for c in (adj_close, sentiment) if c in feature_cols}
    if adj_close not in protected:
        print(f"  ! {adj_close} not found for {ticker}; nothing to protect")

    # Fit the correlation matrix on the training window only (through
    # 2022-12-31), never on either period's test window (2023, 2024).
    train_df = df.loc[df.index <= pd.Timestamp(TRAIN_END)]
    corr = train_df[feature_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    dropped_rows: list[dict] = []
    to_drop: list[str] = []
    for col in upper.columns:
        if col in protected:
            continue
        hits = upper[col][upper[col] >= threshold]
        if not hits.empty:
            kept = hits.index[0]
            to_drop.append(col)
            dropped_rows.append({
                "ticker": ticker,
                "dropped_feature": col,
                "kept_feature": kept,
                "correlation": round(float(hits.iloc[0]), 4),
            })

    # apply the train-fitted drop list to the full dataset (train + both
    # test windows), so every split shares the same reduced column set
    reduced = df.drop(columns=to_drop)
    dst = out_dir / f"{ticker}_reduced.csv"
    reduced.to_csv(dst)

    print(f"[{ticker}] fit on {len(train_df)} training rows (<= {TRAIN_END}); "
          f"{df.shape[1]} -> {reduced.shape[1]} columns "
          f"({len(to_drop)} dropped, target kept: {target}) -> {dst}")
    return dropped_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune redundant (highly correlated) features per ticker.")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                    help="|r| cutoff above which one of a correlated pair is dropped")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dropped: list[dict] = []
    for t in args.tickers:
        all_dropped.extend(prune_ticker(t, out_dir, args.threshold))

    report = pd.DataFrame(all_dropped)
    report_path = out_dir / "dropped_features_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n{len(report)} features dropped across {len(args.tickers)} tickers "
          f"(threshold |r| >= {args.threshold}) -> {report_path}")


if __name__ == "__main__":
    main()
