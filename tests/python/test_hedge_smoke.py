import pandas as pd
import numpy as np

from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge

def test_hedge_smoke_runs():
    # Synthetic minimal market (no download) to ensure the loop runs.
    # NOTE: requires compiled lsm_cpp module to be importable.
    idx = pd.date_range("2023-01-01", periods=40, freq="B")
    close = 100.0 + np.linspace(0, 2, len(idx))
    df = pd.DataFrame({
        "close": close,
        "sigma": 0.2,
        "r": 0.04,
        "q": 0.0,
    }, index=idx)

    cfg = HedgeConfig(
        horizon_days=30,
        tc_bps=0.0,
        engine_cfg={"paths": 5000, "steps": 60, "train_fraction": 0.5, "basis": "Laguerre"}
    )
    path, report = run_single_trade_delta_hedge(df, cfg)

    assert len(path) > 5
    assert "final_pnl" in report
    assert np.isfinite(report["final_pnl"])
