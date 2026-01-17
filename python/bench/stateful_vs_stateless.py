from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import numpy as np

# allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge.data import MarketDataConfig, prepare_market_data
from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge


def run_once(use_stateful: bool) -> float:
    md = prepare_market_data(MarketDataConfig(
        symbol="SPY", start="2022-01-01", end="2022-06-30",
        vol_window=60, rf_mode="constant", rf_constant=0.04, dividend_yield_mode="none"
    ))

    cfg = HedgeConfig(
        horizon_days=60,   # keep this short for benchmarking
        tc_bps=0.0,
        use_stateful_model=use_stateful,
        recalibration_policy="weekly",
        recalibration_days=5,
        engine_cfg={
            "paths": 20_000,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "basis_degree": 2,
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": 123,
        },
        exercise_detection=False,
    )

    t0 = time.perf_counter()
    _path, _report = run_single_trade_delta_hedge(md, cfg)
    t1 = time.perf_counter()
    return t1 - t0


def main():
    t_stateless = run_once(use_stateful=False)
    t_stateful = run_once(use_stateful=True)

    speedup = (t_stateless / t_stateful) if t_stateful > 0 else np.nan
    print(f"Stateless (daily retrain) runtime: {t_stateless:.3f}s")
    print(f"Stateful  (weekly recalib) runtime: {t_stateful:.3f}s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
