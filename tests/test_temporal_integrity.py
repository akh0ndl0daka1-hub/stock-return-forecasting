from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]

walkforward = load_from_path("test_walkforward_module", ROOT / "model" / "LSTM" / "walkforward.py")
missing = load_from_path("test_missing_module", ROOT / "Missing_values" / "handling_ms.py")
feature_reduction = load_from_path("test_feature_reduction_module", ROOT / "feature_reduction" / "feature_reduction.py")
log_returns = load_from_path("test_log_returns_module", ROOT / "data_preprocess" / "feature_engineering" / "log_returns.py")


def _toy_period_fold(extreme_test: bool = False):
    dates = pd.bdate_range("2020-01-01", periods=32).to_numpy(dtype="datetime64[D]")
    target = np.arange(32, dtype=float) / 100.0
    feature = np.linspace(10.0, 20.0, 32)
    if extreme_test:
        feature[20:] = 1000.0
        target[20:] = 10.0
    data = np.column_stack([target, feature])
    prices = 100.0 * np.exp(np.cumsum(target))
    fold = walkforward.build_period_fold(
        dates, prices, data, lookback=3, horizon=2,
        train_start=str(dates[0]), train_end=str(dates[17]),
        test_start=str(dates[18]), test_end=str(dates[29]),
    )
    return dates, data, fold


def test_walkforward_no_lookahead():
    dates, _, fold = _toy_period_fold()
    tr_s, tr_e = fold["train_idx"]
    va_s, va_e = fold["val_idx"]
    te_s, te_e = fold["test_idx"]
    assert tr_s < tr_e <= va_s < va_e <= te_s < te_e
    assert dates[tr_e - 1] < dates[va_s]
    assert dates[va_e - 1] < dates[te_s]


def test_supervised_windows_preserve_temporal_order():
    data = np.arange(1, 7, dtype=float).reshape(-1, 1)
    X, y = walkforward.supervised_window(data, lookback=3, horizon=2)
    assert X.shape == (2, 3, 1)
    assert y.shape == (2, 2)
    assert np.array_equal(X[0, :, 0], [1, 2, 3])
    assert np.array_equal(y[0], [4, 5])
    assert np.array_equal(X[1, :, 0], [2, 3, 4])
    assert np.array_equal(y[1], [5, 6])


def test_forecast_origin_has_no_target_overlap():
    lookback, horizon = 4, 3
    data = np.arange(20, dtype=float).reshape(-1, 1)
    X, y = walkforward.supervised_window(data, lookback, horizon)
    for i in range(len(X)):
        assert X[i, -1, 0] == i + lookback - 1
        assert y[i, 0] == i + lookback
        assert y[i, -1] == i + lookback + horizon - 1
        assert X[i, -1, 0] < y[i, 0]


def test_scaler_fitted_on_training_only():
    _, _, fold = _toy_period_fold(extreme_test=True)
    # Training feature maximum is well below the evaluation-only shock.
    assert fold["feature_scaler"].data_max_[1] < 1000.0
    # Because evaluation values are transformed, not refitted/clipped, they can exceed 1.
    assert float(np.max(fold["X_test"][..., 1])) > 1.0
    assert fold["target_scaler"].data_max_[0] < 10.0


def test_feature_selection_uses_development_data_only(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    out = tmp_path / "out"
    merged.mkdir(); out.mkdir()
    monkeypatch.setattr(feature_reduction, "MERGED_DIR", merged)

    idx_train = pd.date_range("2022-01-01", periods=20, freq="14D")
    idx_eval = pd.date_range("2023-01-02", periods=12, freq="14D")
    idx = idx_train.append(idx_eval)
    a_train = np.arange(len(idx_train), dtype=float)
    # Deliberately unrelated in development data.
    b_train = np.array([0, 7, 1, 9, 2, 8, 3, 6, 4, 5] * 2, dtype=float)
    a_eval = np.arange(100, 100 + len(idx_eval), dtype=float)
    b_eval = a_eval.copy()  # perfectly correlated only in evaluation data
    df = pd.DataFrame({
        "AAPL_feature_A": np.r_[a_train, a_eval],
        "AAPL_feature_B": np.r_[b_train, b_eval],
        "AAPL_Adj_Close": np.linspace(100, 130, len(idx)),
        "AAPL_LOGR_Adj_Close": np.linspace(-0.01, 0.01, len(idx)),
    }, index=idx)
    df.to_csv(merged / "AAPL_merged.csv")
    feature_reduction.prune_ticker("AAPL", out, 0.95)
    reduced = pd.read_csv(out / "AAPL_reduced.csv", index_col=0)
    assert "AAPL_feature_B" in reduced.columns


def test_feature_reduction_preserves_target_and_sentiment(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    out = tmp_path / "out"
    merged.mkdir(); out.mkdir()
    monkeypatch.setattr(feature_reduction, "MERGED_DIR", merged)

    idx = pd.date_range("2022-01-01", periods=20, freq="14D")
    x = np.arange(20, dtype=float)
    df = pd.DataFrame({
        "AAPL_feature_A": x,
        "AAPL_Adj_Close": x,           # protected although perfectly correlated
        "AAPL_sentiment_score": x,     # protected although perfectly correlated
        "AAPL_LOGR_Adj_Close": x / 1000.0,
    }, index=idx)
    df.to_csv(merged / "AAPL_merged.csv")
    feature_reduction.prune_ticker("AAPL", out, 0.95)
    cols = pd.read_csv(out / "AAPL_reduced.csv", nrows=1).columns
    assert "AAPL_LOGR_Adj_Close" in cols
    assert "AAPL_Adj_Close" in cols
    assert "AAPL_sentiment_score" in cols


def test_forward_fill_never_uses_future_values():
    canonical = pd.date_range("2024-01-01", periods=5, freq="D")
    source = pd.DataFrame({"x": [7.0]}, index=[canonical[2]])
    aligned = missing.align_to_calendar(source, canonical)
    assert np.isnan(aligned.loc[canonical[0], "x"])
    assert np.isnan(aligned.loc[canonical[1], "x"])
    assert aligned.loc[canonical[2], "x"] == 7.0
    assert aligned.loc[canonical[4], "x"] == 7.0


def test_sentiment_available_only_on_or_after_publication():
    canonical = pd.date_range("2024-01-01", periods=6, freq="D")
    publication = canonical[3]
    scores = pd.DataFrame({"AAPL_sentiment_score": [0.8]}, index=[publication])
    aligned = missing.align_to_calendar(scores, canonical)
    assert aligned.loc[canonical[:3], "AAPL_sentiment_score"].isna().all()
    assert aligned.loc[publication, "AAPL_sentiment_score"] == 0.8
    assert aligned.loc[canonical[4], "AAPL_sentiment_score"] == 0.8


def test_ewt_future_perturbation_does_not_change_past_features(monkeypatch):
    # causal_band_values only needs the local filter-bank implementation.  A tiny
    # placeholder lets this unit test run even if optional ewtpy is unavailable.
    if "ewtpy" not in sys.modules:
        monkeypatch.setitem(sys.modules, "ewtpy", SimpleNamespace(EWT1D=None))
    ewt = load_from_path("test_ewt_module", ROOT / "data_preprocess" / "feature_engineering" / "ewt.py")
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 60)
    boundaries = np.array([0.45, 0.95, 1.55, 2.25])
    base = ewt.causal_band_values(returns, boundaries, block=16, k=5)
    t = 35
    shocked = returns.copy()
    shocked[t + 1:] += 100.0
    perturbed = ewt.causal_band_values(shocked, boundaries, block=16, k=5)
    assert np.allclose(base[t], perturbed[t], atol=1e-12, rtol=0)


def test_adjusted_log_return_calculation():
    prices = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024-01-01", periods=3))
    r = log_returns.to_log_returns(prices, "AAPL")
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], np.log(1.1))
    assert np.isclose(r.iloc[2], np.log(1.1))


def test_declared_2024_period_is_strict_temporal_holdout():
    train_start, train_end, test_start, test_end = walkforward.PERIODS["2024"]
    assert pd.Timestamp(train_start).year == 2011
    assert pd.Timestamp(train_end).year == 2023
    assert pd.Timestamp(test_start).year == 2024
    assert pd.Timestamp(test_end).year == 2024
    assert pd.Timestamp(train_end) < pd.Timestamp(test_start)
