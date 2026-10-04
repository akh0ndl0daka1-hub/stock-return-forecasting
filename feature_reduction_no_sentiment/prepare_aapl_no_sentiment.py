"""Create the numerical-only AAPL feature file used in the sentiment ablation.

The input is the already reduced sentiment-inclusive AAPL dataset.  Dropping
only ``AAPL_sentiment_score`` ensures that the paired pipelines retain the same
numerical predictors.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_SRC = REPO_ROOT / "feature_reduction" / "AAPL_reduced.csv"
DEFAULT_OUT = HERE / "AAPL_reduced.csv"
SENTIMENT_COLUMN = "AAPL_sentiment_score"


def prepare_no_sentiment(src: Path, out: Path) -> pd.DataFrame:
    """Write the numerical-only AAPL dataset and return the resulting frame."""
    src, out = Path(src), Path(out)
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    if SENTIMENT_COLUMN not in df.columns:
        raise KeyError(
            f"{src} does not contain {SENTIMENT_COLUMN}. Regenerate the reduced AAPL "
            "dataset from the sentiment-inclusive merged data first."
        )
    numerical = df.drop(columns=[SENTIMENT_COLUMN])
    out.parent.mkdir(parents=True, exist_ok=True)
    numerical.to_csv(out)
    return numerical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    before = pd.read_csv(src, index_col=0, parse_dates=True)
    numerical = prepare_no_sentiment(src, out)
    print(f"{src.name}: {before.shape[1]} -> {numerical.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
