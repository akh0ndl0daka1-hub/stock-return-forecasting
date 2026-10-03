"""Run one (ticker, lookback, horizon) TFT forecasting experiment end to end.

The workflow uses two temporally ordered development and evaluation periods.
Hyperparameters are selected across the development periods, after which one final
model is fitted per period and held fixed while forecast origins are swept through
the corresponding evaluation year.
"""

from __future__ import annotations
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import (
    load_and_prepare, period_walk_forward, PERIODS, qrisk,
    compute_interval_metrics, logreturn_to_price, build_fold_rows,
    compute_fold_importance, save_feature_importance_across_folds,
    plot_feature_importance, plot_price_interval, plot_train_results,
    plot_test_results, run_bayes_opt,
    resolve_feature_dir, results_root_for, assert_source_matches,
)
from tft_model import train_tft_model, _resolve_tf_device
from search_space import TFT_SEARCH_SPACE

BASE_DIR = Path(__file__).resolve().parent


def _q_indices(quantiles: list[float]) -> tuple[int, int]:
    return quantiles.index(min(quantiles)), quantiles.index(max(quantiles))


def _hpo_objective_factory(dates_array, prices_array, data, lookback, horizon,
                           tf_device, quantiles, max_epochs, periods=PERIODS):
    def objective(**params) -> float:
        trial_params = {**params, "patience": 5, "batch_size": 32}
        val_losses = []
        for fold in period_walk_forward(
            dates_array, prices_array, data, lookback, horizon, periods
        ):
            n_features = fold["X_train"].shape[-1]
            model, history = train_tft_model(
                fold["X_train"], fold["y_train"], fold["X_val"], fold["y_val"],
                lookback, horizon, n_features, quantiles, trial_params,
                max_epochs=max_epochs, tf_device=tf_device,
            )
            if history.get("val_loss"):
                val_losses.append(min(history["val_loss"]))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
        return float(np.mean(val_losses)) if val_losses else float("inf")
    return objective


def run_config(
    ticker: str,
    lookback: int,
    horizon: int,
    quantiles: list[float] = (0.1, 0.5, 0.9),
    max_epochs: int = 100,
    n_calls: int = 20,
    n_initial_points: int = 5,
    device: str = "cuda",
    out_root: Path | None = None,
    feature_dir: Path | None = None,
) -> dict:
    quantiles = list(quantiles)
    low_idx, high_idx = _q_indices(quantiles)
    low_q, high_q = min(quantiles), max(quantiles)
    nominal_coverage = high_q - low_q
    alpha = 1.0 - nominal_coverage
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    t_start = time.perf_counter()
    dates_array, data, target_col, feature_cols, prices_array, _, _ = load_and_prepare(
        ticker, feature_dir
    )
    n_features = data.shape[1]
    tf_device = _resolve_tf_device(device)

    out_root = Path(out_root) if out_root else results_root_for(feature_dir, BASE_DIR)
    out_dir = out_root / ticker.upper() / f"{lookback}_{horizon}_periods"
    assert_source_matches(out_dir, feature_dir)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Shared Bayesian hyperparameter selection across the two development periods.
    t_hpo_start = time.perf_counter()
    objective = _hpo_objective_factory(
        dates_array, prices_array, data, lookback, horizon, tf_device,
        quantiles, max_epochs=min(max_epochs, 30),
    )
    bo = run_bayes_opt(
        objective, TFT_SEARCH_SPACE, n_calls=n_calls,
        n_initial_points=n_initial_points, random_state=42, verbose=False,
    )
    hpo_seconds = time.perf_counter() - t_hpo_start
    best_params = {**bo.best_params, "patience": 8, "batch_size": 32}
    pd.DataFrame(bo.trials).sort_values("value").to_csv(
        out_dir / "bayes_opt_trials.csv", index=False
    )

    # 2. Fit one final model per temporal period and evaluate every valid origin.
    t_final_start = time.perf_counter()
    test_rows: list[dict] = []
    train_metric_rows: list[dict] = []
    test_metric_rows: list[dict] = []
    importances_by_fold: dict[int, np.ndarray] = {}
    fold_seconds: list[float] = []

    for fold in period_walk_forward(dates_array, prices_array, data, lookback, horizon):
        k = fold["k"]
        period_name = fold["period"]
        t_fold_start = time.perf_counter()

        model, history = train_tft_model(
            fold["X_train"], fold["y_train"], fold["X_val"], fold["y_val"],
            lookback, horizon, n_features, quantiles, best_params,
            max_epochs=max_epochs, tf_device=tf_device,
        )

        # Training metrics are retained in adjusted log-return space.
        y_pred_train_scaled = model.predict(fold["X_train"], verbose=0)
        n_tr = y_pred_train_scaled.shape[0]
        y_pred_train = fold["target_scaler"].inverse_transform(
            y_pred_train_scaled.reshape(-1, 1)
        ).reshape(n_tr, horizon, len(quantiles))
        y_actual_train = fold["y_train_raw"]
        train_qrisk = qrisk(y_actual_train, y_pred_train, quantiles)
        median_err = y_actual_train - y_pred_train[:, :, median_idx]
        train_metric_rows.append({
            "fold": k,
            "period": period_name,
            "n_samples": n_tr,
            "mae": float(np.mean(np.abs(median_err))),
            **{f"qrisk_q{q:.2f}": v for q, v in train_qrisk.items()},
        })

        # Evaluation metrics are also calculated directly in adjusted log-return space.
        y_pred_test_scaled = model.predict(fold["X_test"], verbose=0)
        n_test = y_pred_test_scaled.shape[0]
        y_pred_test = fold["target_scaler"].inverse_transform(
            y_pred_test_scaled.reshape(-1, 1)
        ).reshape(n_test, horizon, len(quantiles))
        y_actual_test = fold["y_test_raw"]

        lower_raw = y_pred_test[:, :, low_idx]
        upper_raw = y_pred_test[:, :, high_idx]
        crossing = lower_raw > upper_raw
        lower = np.minimum(lower_raw, upper_raw)
        upper = np.maximum(lower_raw, upper_raw)
        interval = compute_interval_metrics(y_actual_test, lower, upper, alpha)
        test_qrisk = qrisk(y_actual_test, y_pred_test, quantiles)
        test_mae = float(np.mean(np.abs(y_actual_test - y_pred_test[:, :, median_idx])))
        test_metric_rows.append({
            "fold": k,
            "period": period_name,
            "n_test_windows": n_test,
            "mae": test_mae,
            **interval,
            "crossing_rate": float(np.mean(crossing)),
            **{f"qrisk_q{q:.2f}": v for q, v in test_qrisk.items()},
        })

        # Price paths are reconstructed only for descriptive output/plots.
        anchors = fold["last_prices_test"]
        prices_pred = np.stack([
            logreturn_to_price(y_pred_test[i], anchors[i]) for i in range(n_test)
        ])
        prices_actual = np.stack([
            logreturn_to_price(y_actual_test[i].reshape(horizon, 1), anchors[i]).ravel()
            for i in range(n_test)
        ])

        test_start_i = fold["test_idx"][0]
        for i in range(n_test):
            first_target_i = test_start_i + i
            forecast_origin = np.datetime_as_string(
                dates_array[first_target_i - 1], unit="D"
            )
            target_dates = np.datetime_as_string(
                dates_array[first_target_i:first_target_i + horizon], unit="D"
            )
            test_rows.extend(build_fold_rows(
                k, "test", y_pred_test[i], y_actual_test[i],
                prices_pred[i], prices_actual[i], quantiles, target_dates,
                lookback, horizon, period=period_name,
                forecast_origin=forecast_origin,
            ))

        fold_importance = compute_fold_importance(
            model, fold["X_test"], low_idx, high_idx, tf_device
        )
        if fold_importance is not None:
            importances_by_fold[k] = fold_importance

        model.save(model_dir / f"tft_model_{period_name}.keras")
        fold_seconds.append(time.perf_counter() - t_fold_start)
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    final_seconds = time.perf_counter() - t_final_start
    total_seconds = time.perf_counter() - t_start

    test_pred_df = pd.DataFrame(test_rows)
    train_df = pd.DataFrame(train_metric_rows)
    test_df = pd.DataFrame(test_metric_rows)
    test_pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    train_df.to_csv(out_dir / "train_results.csv", index=False)
    test_df.to_csv(out_dir / "test_results.csv", index=False)

    plot_paths: dict[str, str] = {}
    if importances_by_fold:
        importance_df = save_feature_importance_across_folds(
            importances_by_fold, [target_col] + feature_cols,
            out_dir / "feature_importance_across_folds.csv",
        )
        plot_paths["feature_importance"] = str(plot_feature_importance(
            importance_df, out_dir / "feature_importance_across_folds.png"
        ))
    if len(test_pred_df):
        mid_q = quantiles[median_idx]
        plot_paths["price_interval"] = str(plot_price_interval(
            test_pred_df, out_dir / "price_interval.png",
            low_q, mid_q, high_q, ticker,
        ))
    if len(train_df):
        plot_paths["train_results"] = str(plot_train_results(
            train_df, out_dir / "train_results.png"
        ))
    if len(test_df):
        plot_paths["test_results"] = str(plot_test_results(
            test_df, nominal_coverage, out_dir / "test_results.png"
        ))

    summary = {
        "architecture": "TFT",
        "feature_dir": str(resolve_feature_dir(feature_dir)),
        "ticker": ticker,
        "lookback": lookback,
        "horizon": horizon,
        "fold_mode": "periods",
        "n_folds": len(test_metric_rows),
        "nominal_coverage": nominal_coverage,
        "best_params": best_params,
        "bayes_opt_best_value": bo.best_value,
        "plots": plot_paths,
        "train_overall": {
            "mae": float(train_df["mae"].mean()) if len(train_df) else None,
            **{c: float(train_df[c].mean()) for c in train_df.columns if c.startswith("qrisk_")},
        } if len(train_df) else {},
        "test_overall": {
            "mae": float(test_df["mae"].mean()) if len(test_df) else None,
            "picp": float(test_df["picp"].mean()) if len(test_df) else None,
            "pinaw": float(test_df["pinaw"].mean()) if len(test_df) else None,
            "ais": float(test_df["ais"].mean()) if len(test_df) else None,
            "crossing_rate": float(test_df["crossing_rate"].mean()) if len(test_df) else None,
            **{c: float(test_df[c].mean()) for c in test_df.columns if c.startswith("qrisk_")},
        } if len(test_df) else {},
        "timing_seconds": {
            "hpo": hpo_seconds,
            "final_training": final_seconds,
            "total": total_seconds,
            "mean_per_fold": float(np.mean(fold_seconds)) if fold_seconds else None,
            "per_fold": fold_seconds,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(
        f"[TFT {ticker} {lookback}/{horizon}] periods={len(test_metric_rows)} "
        f"hpo={hpo_seconds:.1f}s final={final_seconds:.1f}s "
        f"total={total_seconds:.1f}s -> {out_dir}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one TFT forecasting configuration.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--lookback", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--n-calls", type=int, default=20)
    ap.add_argument("--n-initial-points", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--feature-dir", default=None)
    args = ap.parse_args()

    quantiles = [float(q) for q in args.quantiles.split(",")]
    if args.lookback <= args.horizon:
        raise ValueError("Experiment requires lookback > horizon")
    run_config(
        args.ticker, args.lookback, args.horizon, quantiles=quantiles,
        max_epochs=args.max_epochs, n_calls=args.n_calls,
        n_initial_points=args.n_initial_points, device=args.device,
        out_root=Path(args.out_root) if args.out_root else None,
        feature_dir=args.feature_dir,
    )


if __name__ == "__main__":
    main()
