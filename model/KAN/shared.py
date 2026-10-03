"""
Single point of coupling to the model-agnostic pipeline.

Everything re-exported here is shared verbatim with the LSTM run -- the same
data loader, the same walk-forward folds, the same quantile loss, the same
interval metrics, the same price reconstruction, the same saliency
attribution, and the same plots. Comparability between architectures depends
on these being the SAME code rather than two copies that can drift, so they
are imported from their existing location instead of being duplicated here.

If these modules are ever moved to a proper `model/common/` package, this is
the only file in KAN/ that needs to change.
"""

from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                  # Masters/model/KAN
_MODEL_DIR = _HERE.parent                                # Masters/model
_LSTM_DIR = _MODEL_DIR / "LSTM"                          # shared pipeline lives here

# KAN's own directory must take priority: LSTM/ also contains search_space.py
# and main.py, which would otherwise shadow this package's versions.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for _p in (str(_MODEL_DIR), str(_LSTM_DIR)):
    if _p not in sys.path:
        sys.path.append(_p)

from data import load_and_prepare                                    # noqa: E402
from walkforward import period_walk_forward, PERIODS                 # noqa: E402
from losses import make_quantile_loss_tf, qrisk                      # noqa: E402
from interval_metrics import compute_interval_metrics                # noqa: E402
from log_return_price import logreturn_to_price                      # noqa: E402
from build_fold_rows import build_fold_rows                          # noqa: E402
from feature_importance import (                                     # noqa: E402
    compute_fold_importance,
    save_feature_importance_across_folds,
)
from plots import (                                                  # noqa: E402
    plot_feature_importance,
    plot_price_interval,
    plot_train_results,
    plot_test_results,
)
from bayes_opt import run_bayes_opt                                  # noqa: E402
from run_paths import (                                             # noqa: E402
    resolve_feature_dir, results_root_for, assert_source_matches,
)

__all__ = [
    "load_and_prepare",
    "period_walk_forward", "PERIODS",
    "make_quantile_loss_tf", "qrisk",
    "compute_interval_metrics",
    "logreturn_to_price",
    "build_fold_rows",
    "compute_fold_importance", "save_feature_importance_across_folds",
    "plot_feature_importance", "plot_price_interval",
    "plot_train_results", "plot_test_results",
    "run_bayes_opt",
    "resolve_feature_dir", "results_root_for", "assert_source_matches",
]
