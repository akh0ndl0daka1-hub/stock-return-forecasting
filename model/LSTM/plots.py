"""Graphical outputs: feature importance, price/interval, train and test results."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path,
                            title: str = "Feature importance (mean across folds)") -> Path:
    """
    importance_df: output of feature_importance.save_feature_importance_across_folds
    (index = feature, has a 'mean' and 'std' column). Plots EVERY feature,
    sorted descending by mean importance -- no top-N truncation.
    """
    df = importance_df.sort_values("mean", ascending=True)  # ascending so largest ends up on top when plotted
    fig_height = max(4, 0.22 * len(df))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(df.index, df["mean"], xerr=df["std"], color="#4C72B0", ecolor="#888888", capsize=2)
    ax.set_xlabel("Mean gradient-based importance (across folds)")
    ax.set_title(f"{title} (n={len(df)} features)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_price_interval(test_predictions: pd.DataFrame, out_path: Path,
                        low_q: float, mid_q: float, high_q: float,
                        ticker: str) -> Path:
    """
    test_predictions: the run's test_predictions.csv (all folds), columns
    include Date, Actual_Price, Pred_Price_Q{low}/{mid}/{high}. Plots the
    actual adjusted close against the median prediction and shaded interval,
    chronologically, de-duplicating overlapping fold predictions per date
    by averaging.
    """
    df = test_predictions.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    low_col = f"Pred_Price_Q{low_q:.2f}"
    mid_col = f"Pred_Price_Q{mid_q:.2f}"
    high_col = f"Pred_Price_Q{high_q:.2f}"

    agg = df.groupby("Date").agg(
        Actual_Price=("Actual_Price", "mean"),
        **{low_col: (low_col, "mean")},
        **{mid_col: (mid_col, "mean")},
        **{high_col: (high_col, "mean")},
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(agg.index, agg["Actual_Price"], label="Actual adjusted close", color="black", linewidth=1.2)
    ax.plot(agg.index, agg[mid_col], label=f"Predicted median (Q{mid_q:.2f})", color="#C44E52", linewidth=1.0)
    ax.fill_between(agg.index, agg[low_col], agg[high_col], color="#C44E52", alpha=0.2,
                    label=f"Predicted interval [Q{low_q:.2f}, Q{high_q:.2f}]")
    ax.set_title(f"{ticker}: actual price vs. predicted interval (evaluation periods)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_train_results(train_results: pd.DataFrame, out_path: Path) -> Path:
    """Per-period training MAE and QRisk, plus their means."""
    qrisk_cols = [c for c in train_results.columns if c.startswith("qrisk_")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(train_results["fold"], train_results["mae"], marker="o", label="MAE")
    ax.axhline(train_results["mae"].mean(), linestyle="--", linewidth=1,
              label=f"mean MAE={train_results['mae'].mean():.4f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Error (log-return space)")
    ax.set_title("Training results: point error per fold")
    ax.legend(fontsize=8)

    ax = axes[1]
    for c in qrisk_cols:
        ax.plot(train_results["fold"], train_results[c], marker="o", label=f"{c} (mean={train_results[c].mean():.3f})")
    ax.set_xlabel("Fold")
    ax.set_ylabel("QRisk")
    ax.set_title("Training results: QRisk per fold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_test_results(test_results: pd.DataFrame, nominal_coverage: float, out_path: Path) -> Path:
    """Per-period evaluation PICP/PINAW/AIS and QRisk."""
    qrisk_cols = [c for c in test_results.columns if c.startswith("qrisk_")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.bar(test_results["fold"], test_results["picp"], color="#55A868")
    ax.axhline(nominal_coverage, color="black", linestyle="--", linewidth=1,
              label=f"nominal={nominal_coverage:.0%}")
    ax.axhline(test_results["picp"].mean(), color="#C44E52", linestyle=":", linewidth=1,
              label=f"mean={test_results['picp'].mean():.0%}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("PICP (coverage)")
    ax.set_title("Evaluation: prediction-interval coverage per fold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(test_results["fold"], test_results["pinaw"], marker="o", label="PINAW")
    ax.plot(test_results["fold"], test_results["ais"], marker="o", label="AIS")
    ax.axhline(test_results["pinaw"].mean(), color="C0", linestyle="--", linewidth=1,
              label=f"mean PINAW={test_results['pinaw'].mean():.3f}")
    ax.axhline(test_results["ais"].mean(), color="C1", linestyle="--", linewidth=1,
              label=f"mean AIS={test_results['ais'].mean():.3f}")
    ax.set_xlabel("Fold")
    ax.set_title("Evaluation: interval width / Winkler score per fold")
    ax.legend(fontsize=8)

    ax = axes[2]
    for c in qrisk_cols:
        ax.plot(test_results["fold"], test_results[c], marker="o", label=f"{c} (mean={test_results[c].mean():.3f})")
    ax.set_xlabel("Fold")
    ax.set_ylabel("QRisk")
    ax.set_title("Evaluation: QRisk per fold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
