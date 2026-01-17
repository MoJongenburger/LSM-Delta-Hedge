import numpy as np
import pandas as pd

from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge


def test_stateful_model_smoke_runs():
    idx = pd.date_range("2023-01-02", periods=25, freq="B")
    df = pd.DataFrame(
        {
            "close": 100.0 + np.linspace(0.0, 1.0, len(idx)),
            "sigma": 0.20,
            "r": 0.04,
            "q": 0.0,
        },
        index=idx,
    )

    cfg = HedgeConfig(
        horizon_days=20,
        tc_bps=0.0,
        use_stateful_model=True,
        recalibration_policy="weekly",
        recalibration_days=5,
        engine_cfg={
            "paths": 5000,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "basis_degree": 2,
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": 123,
            # steps will be overridden internally by remaining_days
        },
        exercise_detection=False,
    )

    path, report = run_single_trade_delta_hedge(df, cfg)
    assert len(path) >= 5
    assert np.isfinite(report["final_pnl"])
