from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from conftest import load_from_path

ROOT = Path(__file__).resolve().parents[1]
losses = None
run_paths = load_from_path("test_run_paths_module", ROOT / "model" / "run_paths.py")
prep_no_sent = load_from_path("test_no_sent_module", ROOT / "feature_reduction_no_sentiment" / "prepare_aapl_no_sentiment.py")


def test_qrisk_known_median_case():
    # Import losses lazily only when TensorFlow is available because the module
    # contains both NumPy QRisk and the TensorFlow training loss.
    import pytest
    pytest.importorskip("tensorflow")
    losses_mod = load_from_path("test_qrisk_loss_module", ROOT / "model" / "LSTM" / "losses.py")
    y = np.array([[1.0, -1.0]])
    pred = np.zeros((1, 2, 1))
    out = losses_mod.qrisk(y, pred, [0.5])
    # Pinball sum = 0.5 + 0.5 = 1; denominator = 2; QRisk = 1.
    assert np.isclose(out[0.5], 1.0)


def test_no_sentiment_dataset_removes_only_sentiment(tmp_path):
    src = tmp_path / "AAPL_reduced.csv"
    dst = tmp_path / "AAPL_reduced_no_sentiment.csv"
    df = pd.DataFrame({
        "AAPL_LOGR_Adj_Close": [0.0, 0.1],
        "AAPL_Adj_Close": [100, 101],
        "AAPL_sentiment_score": [0.2, -0.1],
        "XLK_Close": [50, 51],
    }, index=pd.date_range("2024-01-01", periods=2))
    df.to_csv(src)
    # The repository helper accepts explicit src/dst paths.
    out = prep_no_sent.prepare_no_sentiment(src, dst)
    assert "AAPL_sentiment_score" not in out.columns
    assert set(out.columns) == {"AAPL_LOGR_Adj_Close", "AAPL_Adj_Close", "XLK_Close"}
    assert dst.exists()


def test_results_root_separates_no_sentiment_inputs(tmp_path, monkeypatch):
    normal = tmp_path / "feature_reduction"
    no_sent = tmp_path / "feature_reduction_no_sentiment"
    normal.mkdir(); no_sent.mkdir()
    arch = tmp_path / "model" / "LSTM"
    arch.mkdir(parents=True)
    monkeypatch.setattr(run_paths, "FEATURE_DIR", normal)
    monkeypatch.setattr(run_paths, "NO_SENTIMENT_FEATURE_DIR", no_sent)
    a = run_paths.results_root_for(normal, arch)
    b = run_paths.results_root_for(no_sent, arch)
    assert a.name == "results"
    assert b.name == "results_no_sentiment"
    assert a != b
