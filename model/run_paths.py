"""Repository paths used by the forecasting runners.

Keeping path selection in one module prevents sentiment-inclusive and
numerical-only AAPL experiments from accidentally writing into the same
results tree.
"""

from __future__ import annotations
import re
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent
FEATURE_DIR = REPO_ROOT / "feature_reduction"
NO_SENTIMENT_FEATURE_DIR = REPO_ROOT / "feature_reduction_no_sentiment"


def resolve_feature_dir(feature_dir: str | Path | None = None) -> Path:
    return Path(feature_dir).expanduser().resolve() if feature_dir else FEATURE_DIR.resolve()


def _safe_suffix(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.name).strip("_") or "custom"


def results_root_for(feature_dir: str | Path | None, architecture_dir: str | Path) -> Path:
    """Return the architecture results root appropriate to the feature set."""
    architecture_dir = Path(architecture_dir).resolve()
    resolved = resolve_feature_dir(feature_dir)
    if resolved == FEATURE_DIR.resolve():
        return architecture_dir / "results"
    if resolved == NO_SENTIMENT_FEATURE_DIR.resolve():
        return architecture_dir / "results_no_sentiment"
    return architecture_dir / f"results_{_safe_suffix(resolved)}"


def assert_source_matches(out_dir: str | Path, feature_dir: str | Path | None) -> None:
    """Refuse to mix outputs generated from different feature directories."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = str(resolve_feature_dir(feature_dir))
    marker = out_dir / "_feature_source.txt"
    if marker.exists():
        previous = marker.read_text(encoding="utf-8").strip()
        if previous and previous != source:
            raise RuntimeError(
                f"Output directory {out_dir} already belongs to feature source {previous}; "
                f"requested source is {source}."
            )
    marker.write_text(source + "\n", encoding="utf-8")
