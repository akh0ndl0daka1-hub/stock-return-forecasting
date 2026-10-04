"""Pytest configuration for the examiner-facing verification suite.

The tests use synthetic data and do not download market data or pretrained models.
TensorFlow-only smoke tests are skipped automatically when TensorFlow is not installed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
LSTM = MODEL / "LSTM"

for p in (ROOT, MODEL, LSTM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def rr(repo_root):
    return load_from_path("thesis_reproduce_results", repo_root / "model" / "reproduce_results.py")
