from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _prediction_rows_for_origin(origin, actual, pred, *, horizon=2):
    rows = []
    for step, (y, m) in enumerate(zip(actual, pred), start=1):
        rows.append({
            "Architecture": "LSTM", "Ticker": "AAPL", "Lookback": 5,
            "Horizon": horizon, "Period": "2024", "ForecastOrigin": origin,
            "ForecastStep": step, "Actual_LogReturn": y,
            "Pred_LogReturn_Q0.10": m - 0.01,
            "Pred_LogReturn_Q0.50": m,
            "Pred_LogReturn_Q0.90": m + 0.01,
        })
    return rows


def test_dm_loss_is_aggregated_by_forecast_origin(rr):
    rows = []
    rows += _prediction_rows_for_origin("o1", [0.0, 2.0], [0.0, 0.0])  # errors 0,2 -> mean 1
    rows += _prediction_rows_for_origin("o2", [1.0, 1.0], [0.0, 0.0])  # errors 1,1 -> mean 1
    losses = rr._origin_losses(pd.DataFrame(rows)).sort_values("ForecastOrigin")
    assert len(losses) == 2
    assert np.allclose(losses.model_loss, [1.0, 1.0])


def test_dm_hac_bandwidth_equals_horizon_minus_one(rr):
    assert rr.dm_hac_bandwidth(1) == 0
    assert rr.dm_hac_bandwidth(5) == 4
    assert rr.dm_hac_bandwidth(20) == 19


def test_dm_identical_forecasts_show_no_difference(rr):
    x = np.linspace(0.01, 0.02, 50)
    mean_d, stat, p = rr.dm_test(x, x, horizon=5)
    assert mean_d == 0.0
    assert stat == 0.0
    assert p == 1.0


def test_holm_adjustment_known_values(rr):
    p = np.array([0.01, 0.03, 0.04])
    adjusted = rr.holm_adjust(p)
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_kupiec_nominal_coverage_known_result(rr):
    n, violations, alpha = 100, 20, 0.2
    stat, p = rr.kupiec_test(n, violations, alpha)
    assert np.isclose(stat, 0.0, atol=1e-12)
    assert np.isclose(p, 1.0, atol=1e-12)


def test_kupiec_results_restricted_to_one_day(rr):
    rows = []
    for horizon in (1, 5):
        for i in range(10):
            rows.append({
                "Architecture": "LSTM", "Ticker": "AAPL", "Lookback": 20,
                "Horizon": horizon, "Period": "2024", "ForecastOrigin": f"{horizon}-{i}",
                "Actual_LogReturn": 0.0,
                "Pred_LogReturn_Q0.10": -0.01,
                "Pred_LogReturn_Q0.50": 0.0,
                "Pred_LogReturn_Q0.90": 0.01,
            })
    out = rr.kupiec_results(pd.DataFrame(rows))
    assert len(out) == 1
    assert "Horizon" not in out.columns


def _aapl_metrics(rows):
    return pd.DataFrame(rows, columns=["Architecture", "Ticker", "Lookback", "Horizon", "Period", "mae", "ais", "picp"])


def test_aapl_sentiment_pairs_are_matched_by_configuration(rr):
    sent = _aapl_metrics([
        ["LSTM", "AAPL", 5, 1, "2023", 1.0, 2.0, 0.80],
        ["KAN", "AAPL", 20, 5, "2023", 3.0, 4.0, 0.70],
    ])
    # Reverse order deliberately; merge must use keys, not row order.
    num = _aapl_metrics([
        ["KAN", "AAPL", 20, 5, "2023", 3.5, 4.5, 0.72],
        ["LSTM", "AAPL", 5, 1, "2023", 1.5, 2.5, 0.78],
    ])
    paired = rr.pair_aapl_metrics(sent, num).sort_values("Architecture")
    assert len(paired) == 2
    kan = paired[paired.Architecture == "KAN"].iloc[0]
    assert kan.mae_sentiment == 3.0 and kan.mae_numerical == 3.5


def test_aapl_difference_direction_negative_means_improvement(rr):
    sent = _aapl_metrics([
        ["LSTM", "AAPL", 5, 1, "2023", 1.0, 2.0, 0.79],
        ["LSTM", "AAPL", 5, 1, "2024", 1.0, 2.0, 0.79],
    ])
    num = _aapl_metrics([
        ["LSTM", "AAPL", 5, 1, "2023", 1.5, 2.5, 0.75],
        ["LSTM", "AAPL", 5, 1, "2024", 1.5, 2.5, 0.75],
    ])
    paired = rr.pair_aapl_metrics(sent, num)
    assert (paired.mae_sentiment - paired.mae_numerical).iloc[0] < 0
    assert (paired.ais_sentiment - paired.ais_numerical).iloc[0] < 0
    assert (paired.abs_coverage_error_sentiment - paired.abs_coverage_error_numerical).iloc[0] < 0


def test_bootstrap_reproducibility(rr):
    x = [-0.3, -0.1, 0.2, 0.4, -0.2]
    a = rr.bootstrap_mean_interval(x, n_boot=2000, seed=42)
    b = rr.bootstrap_mean_interval(x, n_boot=2000, seed=42)
    assert a == b


def test_temporal_stability_matches_same_configuration(rr):
    rows = []
    for arch, base in [("LSTM", 1.0), ("KAN", 2.0), ("TFT", 3.0)]:
        for year, factor in [("2023", 1.0), ("2024", 1.1)]:
            rows.append({"Architecture": arch, "Ticker": "AAPL", "Lookback": 5,
                         "Horizon": 1, "Period": year, "mae": base*factor,
                         "zero_mae": 4.0, "picp": .8, "pinaw": .3, "ais": base*factor})
    metrics = pd.DataFrame(rows).sample(frac=1.0, random_state=7).reset_index(drop=True)
    temporal, winners = rr.temporal_and_winner_summaries(metrics)
    assert temporal["same_lowest_mae"] == 1
    assert temporal["same_lowest_ais"] == 1
    assert winners.iloc[0].MAE_winner == "LSTM"


def test_symmetric_relative_change(rr):
    out = rr.symmetric_relative_change(np.array([1.0]), np.array([1.2]))
    assert np.isclose(out[0], 0.2 / 1.1)
    # Symmetric: reversing the years gives the same magnitude.
    rev = rr.symmetric_relative_change(np.array([1.2]), np.array([1.0]))
    assert np.isclose(out[0], rev[0])
