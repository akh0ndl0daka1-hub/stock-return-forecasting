"""Reconstruct comparative and statistical results from saved model outputs.

This script does not refit forecasting models.  It reads the retained
``test_predictions.csv``, feature-attribution files, Bayesian optimisation
summaries, and timing metadata, then reconstructs the principal Chapter 6
analyses in adjusted log-return space.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr, t as student_t, wilcoxon

ARCHITECTURES = ("LSTM", "KAN", "TFT")
QUANTILES = (0.10, 0.50, 0.90)
ALPHA = 0.20
NOMINAL_COVERAGE = 0.80


def dm_hac_bandwidth(horizon: int) -> int:
    """Bartlett HAC bandwidth used for an H-step forecast comparison."""
    return max(int(horizon) - 1, 0)


def symmetric_relative_change(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Absolute change relative to the mean of two positive comparison values."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    denom = (a + b) / 2.0
    return np.divide(np.abs(b - a), denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)


def bootstrap_mean_interval(values: Iterable[float], *, n_boot: int = 10000,
                            seed: int = 42, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for a paired mean difference."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        raise ValueError("values must not be empty")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(int(n_boot), x.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail)))


def _pred_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    return ("Pred_LogReturn_Q0.10", "Pred_LogReturn_Q0.50", "Pred_LogReturn_Q0.90")


def _parse_config_dir(path: Path) -> tuple[int, int]:
    stem = path.name.replace("_periods", "")
    lb, h = stem.split("_")[:2]
    return int(lb), int(h)



def discover_predictions(repo_root: Path, result_dir_name: str = "results") -> pd.DataFrame:
    """Load every architecture/configuration prediction file into one table."""
    frames = []
    for arch in ARCHITECTURES:
        base = repo_root / "model" / arch / result_dir_name
        if not base.exists():
            continue
        for csv_path in sorted(base.glob("*/*_periods/test_predictions.csv")):
            ticker = csv_path.parent.parent.name.upper()
            lookback, horizon = _parse_config_dir(csv_path.parent)
            df = pd.read_csv(csv_path)
            required = {"Actual_LogReturn", "Pred_LogReturn_Q0.10",
                        "Pred_LogReturn_Q0.50", "Pred_LogReturn_Q0.90"}
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"{csv_path}: missing columns {sorted(missing)}")
            if "TargetDate" not in df.columns:
                if "Date" not in df.columns:
                    raise KeyError(f"{csv_path}: requires TargetDate or Date")
                df["TargetDate"] = df["Date"]
            if "ForecastStep" not in df.columns:
                df["ForecastStep"] = df.groupby(["Fold"] if "Fold" in df.columns else lambda x: 0).cumcount() + 1
            if "ForecastOrigin" not in df.columns:
                target = pd.to_datetime(df["TargetDate"])
                # For legacy output, identify origins by each sequential block of H rows.
                block = np.arange(len(df)) // horizon
                df["ForecastOrigin"] = block.astype(str)
            if "Period" not in df.columns or df["Period"].isna().all():
                df["Period"] = pd.to_datetime(df["TargetDate"]).dt.year.astype(str)
            df["Architecture"] = arch
            df["Ticker"] = ticker
            df["Lookback"] = lookback
            df["Horizon"] = horizon
            frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No saved predictions found under {repo_root}/model/<architecture>/{result_dir_name}/"
        )
    out = pd.concat(frames, ignore_index=True)
    out["Period"] = out["Period"].astype(str)
    return out


def interval_metrics_from_rows(df: pd.DataFrame) -> dict[str, float]:
    low_col, mid_col, high_col = _pred_cols(df)
    y = df["Actual_LogReturn"].to_numpy(float)
    low_raw = df[low_col].to_numpy(float)
    high_raw = df[high_col].to_numpy(float)
    low = np.minimum(low_raw, high_raw)
    high = np.maximum(low_raw, high_raw)
    inside = (y >= low) & (y <= high)
    width = high - low
    rng = float(np.max(y) - np.min(y))
    pinaw = float(np.mean(width) / rng) if rng > 0 else 0.0
    score = width.copy()
    below, above = y < low, y > high
    score[below] += (2.0 / ALPHA) * (low[below] - y[below])
    score[above] += (2.0 / ALPHA) * (y[above] - high[above])
    median = df[mid_col].to_numpy(float)
    return {
        "mae": float(np.mean(np.abs(y - median))),
        "zero_mae": float(np.mean(np.abs(y))),
        "picp": float(np.mean(inside)),
        "coverage_deviation": float(np.mean(inside) - NOMINAL_COVERAGE),
        "pinaw": pinaw,
        "ais": float(np.mean(score)),
        "crossing_rate": float(np.mean(low_raw > high_raw)),
    }


def configuration_period_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    keys = ["Architecture", "Ticker", "Lookback", "Horizon", "Period"]
    rows = []
    for key, grp in pred.groupby(keys, sort=True):
        rows.append(dict(zip(keys, key)) | interval_metrics_from_rows(grp))
    return pd.DataFrame(rows)


def horizon_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (metrics.groupby(["Architecture", "Horizon"], as_index=False)
            .agg(mae=("mae", "mean"), zero_mae=("zero_mae", "mean"),
                 coverage_deviation=("coverage_deviation", "mean"),
                 pinaw=("pinaw", "mean"), ais=("ais", "mean"),
                 crossing_rate=("crossing_rate", "mean"))
            .assign(relative_mae=lambda d: d.mae / d.zero_mae))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int) -> tuple[float, float, float]:
    """Two-sided Diebold-Mariano test with Bartlett HAC and HLN correction."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    if a.shape != b.shape:
        raise ValueError("DM loss series must be matched")
    d = a - b
    n = len(d)
    if n < 3:
        return float(np.mean(d)), float("nan"), float("nan")
    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    bandwidth = dm_hac_bandwidth(horizon)
    lrv = gamma0
    for lag in range(1, min(bandwidth, n - 1) + 1):
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (bandwidth + 1.0)
        lrv += 2.0 * weight * gamma
    if lrv <= 0:
        if mean_d == 0:
            return mean_d, 0.0, 1.0
        return mean_d, math.copysign(float("inf"), mean_d), 0.0
    dm = mean_d / math.sqrt(lrv / n)
    h = int(horizon)
    factor_sq = (n + 1 - 2 * h + (h * (h - 1) / n)) / n
    factor = math.sqrt(max(factor_sq, 0.0))
    dm_hln = dm * factor
    p = float(2.0 * student_t.sf(abs(dm_hln), df=n - 1))
    return mean_d, float(dm_hln), p


def _origin_losses(df: pd.DataFrame) -> pd.DataFrame:
    _, mid_col, _ = _pred_cols(df)
    tmp = df.copy()
    tmp["model_abs_error"] = (tmp["Actual_LogReturn"] - tmp[mid_col]).abs()
    tmp["zero_abs_error"] = tmp["Actual_LogReturn"].abs()
    keys = ["Architecture", "Ticker", "Lookback", "Horizon", "Period", "ForecastOrigin"]
    return (tmp.groupby(keys, as_index=False)
            .agg(model_loss=("model_abs_error", "mean"),
                 zero_loss=("zero_abs_error", "mean")))


def dm_vs_zero(pred: pd.DataFrame) -> pd.DataFrame:
    losses = _origin_losses(pred)
    rows = []
    group_keys = ["Architecture", "Ticker", "Lookback", "Horizon", "Period"]
    for key, grp in losses.groupby(group_keys, sort=True):
        mean_d, stat, p = dm_test(grp["model_loss"], grp["zero_loss"], int(key[3]))
        rows.append(dict(zip(group_keys, key)) | {
            "mean_loss_difference": mean_d, "dm_stat": stat, "p": p,
            "model_lower_loss": mean_d < 0,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for arch, idx in out.groupby("Architecture").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p"].to_numpy())
    out["significant"] = out["p_holm"] < 0.05
    out["outcome"] = np.select(
        [out.significant & (out.mean_loss_difference < 0),
         out.significant & (out.mean_loss_difference > 0)],
        ["model_significantly_lower", "benchmark_significantly_lower"],
        default="no_significant_difference",
    )
    return out


def dm_pairwise(pred: pd.DataFrame) -> pd.DataFrame:
    losses = _origin_losses(pred)
    pairs = [("LSTM", "KAN"), ("LSTM", "TFT"), ("KAN", "TFT")]
    rows = []
    key_cfg = ["Ticker", "Lookback", "Horizon", "Period"]
    for a, b in pairs:
        la = losses[losses.Architecture == a]
        lb = losses[losses.Architecture == b]
        for cfg, ga in la.groupby(key_cfg, sort=True):
            gb = lb
            for col, val in zip(key_cfg, cfg):
                gb = gb[gb[col] == val]
            merged = ga.merge(gb, on=[*key_cfg, "ForecastOrigin"], suffixes=("_a", "_b"))
            if merged.empty:
                continue
            mean_d, stat, p = dm_test(merged.model_loss_a, merged.model_loss_b, int(cfg[2]))
            rows.append(dict(zip(key_cfg, cfg)) | {
                "first": a, "second": b, "mean_loss_difference": mean_d,
                "dm_stat": stat, "p": p, "first_lower_loss": mean_d < 0,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for pair, idx in out.groupby(["first", "second"]).groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p"].to_numpy())
    out["significant"] = out.p_holm < 0.05
    out["outcome"] = np.select(
        [out.significant & (out.mean_loss_difference < 0),
         out.significant & (out.mean_loss_difference > 0)],
        ["first_significantly_lower", "second_significantly_lower"],
        default="no_significant_difference",
    )
    return out


def kupiec_test(n: int, violations: int, alpha: float = ALPHA) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    x = int(violations)
    phat = x / n
    def term(count: int, prob: float) -> float:
        if count == 0:
            return 0.0
        if prob <= 0.0:
            return float("-inf")
        return count * math.log(prob)
    log_l0 = term(n - x, 1 - alpha) + term(x, alpha)
    log_l1 = term(n - x, 1 - phat) + term(x, phat)
    lr = -2.0 * (log_l0 - log_l1)
    return float(lr), float(chi2.sf(lr, df=1))


def kupiec_results(pred: pd.DataFrame) -> pd.DataFrame:
    low_col, _, high_col = _pred_cols(pred)
    one = pred[pred.Horizon == 1].copy()
    rows = []
    keys = ["Architecture", "Ticker", "Lookback", "Period"]
    for key, grp in one.groupby(keys, sort=True):
        low = np.minimum(grp[low_col].to_numpy(float), grp[high_col].to_numpy(float))
        high = np.maximum(grp[low_col].to_numpy(float), grp[high_col].to_numpy(float))
        y = grp.Actual_LogReturn.to_numpy(float)
        violations = int(np.sum((y < low) | (y > high)))
        stat, p = kupiec_test(len(y), violations)
        rows.append(dict(zip(keys, key)) | {
            "n": len(y), "violations": violations, "lr_uc": stat, "p": p,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = np.nan
    for arch, idx in out.groupby("Architecture").groups.items():
        out.loc[idx, "p_holm"] = holm_adjust(out.loc[idx, "p"].to_numpy())
    out["reject_nominal_coverage"] = out.p_holm < 0.05
    return out


def pair_aapl_metrics(sentiment_metrics: pd.DataFrame, numerical_metrics: pd.DataFrame) -> pd.DataFrame:
    """Match period-averaged AAPL results by architecture, lookback, and horizon."""
    avg_keys = ["Architecture", "Ticker", "Lookback", "Horizon"]
    mf = sentiment_metrics.copy()
    mn = numerical_metrics.copy()
    mf["abs_coverage_error"] = (mf["picp"] - NOMINAL_COVERAGE).abs()
    mn["abs_coverage_error"] = (mn["picp"] - NOMINAL_COVERAGE).abs()
    aggf = mf.groupby(avg_keys, as_index=False).agg(
        mae=("mae", "mean"), ais=("ais", "mean"),
        abs_coverage_error=("abs_coverage_error", "mean"))
    aggn = mn.groupby(avg_keys, as_index=False).agg(
        mae=("mae", "mean"), ais=("ais", "mean"),
        abs_coverage_error=("abs_coverage_error", "mean"))
    return aggf.merge(aggn, on=avg_keys, suffixes=("_sentiment", "_numerical"), validate="one_to_one")


def summarise_aapl_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon, Holm and bootstrap summaries for the paired AAPL comparison."""
    rows = []
    for metric in ["mae", "ais", "abs_coverage_error"]:
        values = (paired[f"{metric}_sentiment"] - paired[f"{metric}_numerical"]).to_numpy(float)
        if len(values) == 0:
            raise ValueError("No matched AAPL configurations were supplied")
        stat, p = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        ci_low, ci_high = bootstrap_mean_interval(values, n_boot=10000, seed=42)
        rows.append({
            "outcome": metric,
            "mean_difference": float(values.mean()),
            "improved_pairs": int(np.sum(values < 0)),
            "n_pairs": len(values),
            "wilcoxon_stat": float(stat),
            "p": float(p),
            "bootstrap_2.5": ci_low,
            "bootstrap_97.5": ci_high,
        })
    summary = pd.DataFrame(rows)
    summary["p_holm"] = holm_adjust(summary.p)
    return summary


def aapl_sentiment_comparison(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = discover_predictions(repo_root, "results")
    no_sent = discover_predictions(repo_root, "results_no_sentiment")
    full = full[full.Ticker == "AAPL"]
    no_sent = no_sent[no_sent.Ticker == "AAPL"]
    mf = configuration_period_metrics(full)
    mn = configuration_period_metrics(no_sent)
    paired = pair_aapl_metrics(mf, mn)
    return paired, summarise_aapl_pairs(paired)


def temporal_and_winner_summaries(metrics: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    wide = metrics.pivot_table(index=["Architecture", "Ticker", "Lookback", "Horizon"],
                               columns="Period", values=["mae", "picp", "pinaw", "ais", "zero_mae"])
    years = [c for c in ["2023", "2024"] if ("mae", c) in wide.columns]
    temporal = {}
    if len(years) == 2:
        y1, y2 = years
        rho, p = spearmanr(wide[("mae", y1)], wide[("mae", y2)])
        rel = symmetric_relative_change(wide[("mae", y1)].to_numpy(float), wide[("mae", y2)].to_numpy(float))
        temporal = {
            "years": [y1, y2], "spearman_mae": float(rho), "spearman_p": float(p),
            "median_symmetric_relative_mae_change": float(np.median(rel)),
            "p90_symmetric_relative_mae_change": float(np.quantile(rel, 0.9)),
        }
        for metric in ["mae", "zero_mae", "picp", "pinaw", "ais"]:
            temporal[f"mean_{metric}_{y1}"] = float(wide[(metric, y1)].mean())
            temporal[f"mean_{metric}_{y2}"] = float(wide[(metric, y2)].mean())
        temporal[f"relative_mae_{y1}"] = temporal[f"mean_mae_{y1}"] / temporal[f"mean_zero_mae_{y1}"]
        temporal[f"relative_mae_{y2}"] = temporal[f"mean_mae_{y2}"] / temporal[f"mean_zero_mae_{y2}"]

        # Same architecture records the minimum in both years, per ticker/L/H.
        same = {"mae": 0, "ais": 0}
        n_cfg = 0
        for cfg, grp in metrics.groupby(["Ticker", "Lookback", "Horizon"]):
            if set(grp.Period) >= {y1, y2}:
                n_cfg += 1
                for metric in ["mae", "ais"]:
                    a1 = grp[grp.Period == y1].sort_values(metric).iloc[0].Architecture
                    a2 = grp[grp.Period == y2].sort_values(metric).iloc[0].Architecture
                    same[metric] += int(a1 == a2)
        temporal["same_lowest_mae"] = same["mae"]
        temporal["same_lowest_ais"] = same["ais"]
        temporal["winner_persistence_denominator"] = n_cfg

    period_avg = (metrics.groupby(["Architecture", "Ticker", "Lookback", "Horizon"], as_index=False)
                  .agg(mae=("mae", "mean"), ais=("ais", "mean")))
    rows = []
    for (ticker, lb, h), grp in period_avg.groupby(["Ticker", "Lookback", "Horizon"]):
        rows.append({"Ticker": ticker, "Lookback": lb, "Horizon": h,
                     "MAE_winner": grp.sort_values("mae").iloc[0].Architecture,
                     "AIS_winner": grp.sort_values("ais").iloc[0].Architecture})
    winner_detail = pd.DataFrame(rows)
    return temporal, winner_detail


def _feature_family(feature: str, ticker: str) -> str:
    f = feature.upper()
    if "SENTIMENT" in f:
        return "AAPL sentiment"
    if "_EWT_BAND" in f:
        return "EWT components"
    if f == f"{ticker}_LOGR_ADJ_CLOSE".upper():
        return "Recent target adjusted log return"
    technical_tokens = ("_SMA_", "_EMA_", "_MACD", "_RSI_", "_ROC_", "_CCI_", "_WILLR_", "_ATR_", "_BB_", "_AD", "_OBV")
    if any(tok in f for tok in technical_tokens):
        return "Technical indicators"
    if f.startswith("SPY_"):
        return "SPY variables"
    if any(f.startswith(prefix) for prefix in ("XLK_", "XLF_", "XLV_", "XLE_", "XLI_", "XLY_")):
        return "Sector ETF variables"
    if any(f.startswith(prefix) for prefix in ("VIX_", "TREASURY_20Y_", "TREASURY_1_3Y_", "GOLD_", "US_DOLLAR_INDEX_", "CRUDE_OIL_", "COPPER_")):
        return "Macroeconomic and cross-asset indicators"
    valuation_tokens = ("PRICE_TO_", "PE_GROWTH", "ENTERPRISE_VALUE", "FREE_CASH_FLOW_YIELD", "DIVIDEND_YIELD")
    if any(tok in f for tok in valuation_tokens):
        return "Valuation ratios"
    if f.startswith(ticker.upper() + "_"):
        return "Company price and volume"
    return "Other"


def attribution_summary(repo_root: Path) -> pd.DataFrame:
    records = []
    for arch in ARCHITECTURES:
        base = repo_root / "model" / arch / "results"
        for path in sorted(base.glob("*/*_periods/feature_importance_across_folds.csv")):
            ticker = path.parent.parent.name.upper()
            lb, h = _parse_config_dir(path.parent)
            df = pd.read_csv(path, index_col=0)
            fold_cols = [c for c in df.columns if c.startswith("fold_")]
            for col in fold_cols:
                vals = df[col].astype(float)
                denom = vals.sum()
                if denom <= 0:
                    continue
                shares = vals / denom
                tmp = pd.DataFrame({"feature": df.index.astype(str), "share": shares.to_numpy()})
                tmp["family"] = tmp.feature.map(lambda x: _feature_family(x, ticker))
                fam = tmp.groupby("family", as_index=False).share.sum()
                period = "2023" if col == "fold_0" else "2024" if col == "fold_1" else col
                for _, row in fam.iterrows():
                    records.append({"Architecture": arch, "Ticker": ticker, "Lookback": lb,
                                    "Horizon": h, "Period": period,
                                    "Family": row.family, "Share": row.share})
    if not records:
        return pd.DataFrame(columns=["Family", "mean_share"])
    detail = pd.DataFrame(records)
    return detail.groupby("Family", as_index=False).agg(mean_share=("Share", "mean"))\
                 .sort_values("mean_share", ascending=False)


def selected_hyperparameters(repo_root: Path) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        base = repo_root / "model" / arch / "results"
        for path in sorted(base.glob("*/*_periods/summary.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            lb, h = _parse_config_dir(path.parent)
            row = {"Architecture": arch, "Ticker": path.parent.parent.name.upper(),
                   "Lookback": lb, "Horizon": h,
                   "validation_objective": obj.get("bayes_opt_best_value")}
            row.update({k: v for k, v in obj.get("best_params", {}).items()
                        if k not in {"patience", "batch_size"}})
            rows.append(row)
    return pd.DataFrame(rows)


def runtime_summary(repo_root: Path) -> pd.DataFrame:
    rows = []
    for arch in ARCHITECTURES:
        base = repo_root / "model" / arch / "results"
        for path in sorted(base.glob("*/*_periods/summary.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            lb, h = _parse_config_dir(path.parent)
            rows.append({"Architecture": arch, "Ticker": path.parent.parent.name.upper(),
                         "Lookback": lb, "Horizon": h,
                         "hours": float(obj.get("timing_seconds", {}).get("total", np.nan)) / 3600.0})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    by_h = detail.groupby(["Architecture", "Horizon"], as_index=False).agg(mean_hours=("hours", "mean"))
    overall = detail.groupby("Architecture", as_index=False).agg(mean_hours=("hours", "mean"))
    overall["Horizon"] = "Overall"
    return pd.concat([by_h, overall], ignore_index=True)


def verification_summary(out_dir: Path, reference_path: Path) -> dict:
    """Compare reconstructed aggregate tables with archived reference values."""
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    report = {"reference": str(reference_path), "checks": []}
    hs = pd.read_csv(out_dir / "horizon_summary.csv")
    for arch, horizons in ref["horizon_summary"].items():
        for h_str, expected in horizons.items():
            row = hs[(hs.Architecture == arch) & (hs.Horizon.astype(str) == str(h_str))]
            if row.empty:
                report["checks"].append({"name": f"{arch} H{h_str}", "status": "missing"})
                continue
            row = row.iloc[0]
            for metric, exp in expected.items():
                got = float(row[metric])
                tol = 5e-6 if metric not in {"pinaw", "coverage_deviation", "relative_mae"} else 5e-4
                report["checks"].append({"name": f"{arch} H{h_str} {metric}",
                                         "expected": exp, "actual": got,
                                         "status": "pass" if abs(got-exp) <= tol else "different"})
    report["all_pass"] = all(c["status"] == "pass" for c in report["checks"])
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-reference", action="store_true")
    args = ap.parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out).resolve() if args.out else repo_root / "reproduced_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = discover_predictions(repo_root, "results")
    metrics = configuration_period_metrics(pred)
    metrics.to_csv(out_dir / "config_period_metrics.csv", index=False)
    horizon_summary(metrics).to_csv(out_dir / "horizon_summary.csv", index=False)
    dm_vs_zero(pred).to_csv(out_dir / "dm_vs_zero.csv", index=False)
    dm_pairwise(pred).to_csv(out_dir / "dm_pairwise.csv", index=False)
    kupiec_results(pred).to_csv(out_dir / "kupiec.csv", index=False)

    temporal, winner_detail = temporal_and_winner_summaries(metrics)
    (out_dir / "temporal_stability.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")
    winner_detail.to_csv(out_dir / "winner_detail.csv", index=False)
    winner_counts = pd.concat([
        winner_detail.groupby(["Ticker", "MAE_winner"]).size().unstack(fill_value=0).add_prefix("MAE_") ,
        winner_detail.groupby(["Ticker", "AIS_winner"]).size().unstack(fill_value=0).add_prefix("AIS_")
    ], axis=1).reset_index()
    winner_counts.to_csv(out_dir / "winner_counts_by_company.csv", index=False)

    attribution_summary(repo_root).to_csv(out_dir / "attribution_family_summary.csv", index=False)
    selected_hyperparameters(repo_root).to_csv(out_dir / "selected_hyperparameters.csv", index=False)
    runtime_summary(repo_root).to_csv(out_dir / "runtime_summary.csv", index=False)

    try:
        paired, sentiment = aapl_sentiment_comparison(repo_root)
        paired.to_csv(out_dir / "aapl_sentiment_pairs.csv", index=False)
        sentiment.to_csv(out_dir / "aapl_sentiment_summary.csv", index=False)
    except FileNotFoundError:
        print("Numerical-only AAPL results not found; skipped sentiment comparison.")

    if args.verify_reference:
        reference = repo_root / "reference" / "reported_results.json"
        report = verification_summary(out_dir, reference)
        (out_dir / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Reference verification: {'PASS' if report['all_pass'] else 'CHECK DIFFERENCES'}")

    print(f"Reproduced analyses written to {out_dir}")


if __name__ == "__main__":
    main()
