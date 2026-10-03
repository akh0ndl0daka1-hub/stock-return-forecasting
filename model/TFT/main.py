"""Run the complete forecasting experiment grid for TFT."""

from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_paths import results_root_for

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_tft.py"
TICKERS = ["AAPL", "BA", "JNJ", "JPM", "NKE", "XOM"]
LOOKBACKS = [5, 20, 60]
HORIZONS = [1, 5, 20]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TFT forecasting grid.")
    ap.add_argument("--tickers", nargs="*", default=TICKERS)
    ap.add_argument("--lookbacks", nargs="*", type=int, default=LOOKBACKS)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--quantiles", default="0.1,0.5,0.9")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--n-calls", type=int, default=20)
    ap.add_argument("--n-initial-points", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--feature-dir", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else results_root_for(args.feature_dir, HERE)
    combos = [(t, lb, h) for t in args.tickers for lb in args.lookbacks
              for h in args.horizons if lb > h]
    print(f"{len(combos)} combinations to run (lookback > horizon only)")

    log_rows = []
    for ticker, lookback, horizon in combos:
        cmd = [
            sys.executable, str(RUNNER), "--ticker", ticker,
            "--lookback", str(lookback), "--horizon", str(horizon),
            "--quantiles", args.quantiles, "--max-epochs", str(args.max_epochs),
            "--n-calls", str(args.n_calls),
            "--n-initial-points", str(args.n_initial_points),
            "--device", args.device, "--out-root", str(out_root),
        ]
        if args.feature_dir:
            cmd += ["--feature-dir", args.feature_dir]

        print(f"\n=== TFT {ticker} lookback={lookback} horizon={horizon} ===")
        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = time.perf_counter() - t0
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        log_rows.append({
            "ticker": ticker, "lookback": lookback, "horizon": horizon,
            "status": status, "seconds": round(elapsed, 1),
            "out_dir": str(out_root / ticker / f"{lookback}_{horizon}_periods"),
        })
        if result.returncode != 0:
            print(f"Stopping after failure in {ticker} {lookback}/{horizon}")
            break

    log_df = pd.DataFrame(log_rows)
    out_root.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out_root / "run_log.csv", index=False)
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()
