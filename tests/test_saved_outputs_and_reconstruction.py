from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]
build_rows_mod = load_from_path("test_build_rows_module", ROOT / "model" / "LSTM" / "build_fold_rows.py")


def test_saved_prediction_schema():
    pred = np.array([
        [-0.02, -0.01, 0.00],
        [-0.01, 0.00, 0.01],
    ])
    actual = np.array([-0.01, 0.005])
    price_pred = np.array([[98, 99, 100], [99, 100, 101]], dtype=float)
    price_actual = np.array([99, 100], dtype=float)
    rows = build_rows_mod.build_fold_rows(
        0, "test", pred, actual, price_pred, price_actual,
        [0.1, 0.5, 0.9], np.array(["2024-01-03", "2024-01-04"]),
        5, 2, period="2024", forecast_origin="2024-01-02")
    df = pd.DataFrame(rows)
    required = {
        "Period", "ForecastOrigin", "ForecastStep", "TargetDate",
        "Actual_LogReturn", "Pred_LogReturn_Q0.10", "Pred_LogReturn_Q0.50",
        "Pred_LogReturn_Q0.90", "QuantileCrossing",
    }
    assert required.issubset(df.columns)
    assert list(df.ForecastStep) == [1, 2]
    assert (df.ForecastOrigin == "2024-01-02").all()


def test_result_reconstruction_from_saved_predictions(rr):
    df = pd.DataFrame({
        "Architecture": "LSTM", "Ticker": "AAPL", "Lookback": 5,
        "Horizon": 1, "Period": "2024",
        "ForecastOrigin": ["o1", "o2", "o3", "o4"],
        "Actual_LogReturn": [-0.02, -0.01, 0.01, 0.02],
        "Pred_LogReturn_Q0.10": [-0.03, -0.02, 0.0, 0.01],
        "Pred_LogReturn_Q0.50": [-0.02, 0.0, 0.0, 0.02],
        "Pred_LogReturn_Q0.90": [-0.01, 0.0, 0.02, 0.03],
    })
    metrics = rr.configuration_period_metrics(df)
    assert len(metrics) == 1
    row = metrics.iloc[0]
    expected_mae = np.mean([0.0, 0.01, 0.01, 0.0])
    assert np.isclose(row.mae, expected_mae)
    assert np.isclose(row.zero_mae, np.mean(np.abs(df.Actual_LogReturn)))
    assert row.picp == 1.0
    assert row.crossing_rate == 0.0
