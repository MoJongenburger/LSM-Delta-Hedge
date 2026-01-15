import numpy as np
import pandas as pd

from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge


def _synthetic_market(n=40):
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100.0 + np.linspace(0.0, 1.0, n)  # gentle drift
    df = pd.DataFrame(
        {
            "close": close,
            "sigma": 0.20,
            "r": 0.04,
            "q": 0.0,
        },
        index=idx,
    )
    return df


def test_hedge_atm_no_tc_smoke():
    md = _synthetic_market(35)
    cfg = HedgeConfig(
        horizon_days=25,
        tc_bps=0.0,
        engine_cfg={
            "steps": 60,
            "paths": 5000,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "seed": 123,
        },
        exercise_detection=False,  # remove exercise heuristic from this test
        use_cache=False,
    )
    path, report = run_single_trade_delta_hedge(md, cfg)
    assert len(path) >= 10
    assert np.isfinite(report["final_pnl"])


def test_hedge_exercise_detection_triggers_deep_itm():
    # Deep ITM put: set K fixed high relative to S
    md = _synthetic_market(40)
    # Make underlying fall so payoff becomes large and time value collapses
    md["close"] = np.linspace(120.0, 80.0, len(md))

    cfg = HedgeConfig(
        horizon_days=30,
        strike_mode="fixed",
        fixed_strike=120.0,
        tc_bps=0.0,
        engine_cfg={
            "steps": 60,
            "paths": 8000,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "seed": 123,
        },
        exercise_detection=True,
        exercise_k_stderr=3.0,
        exercise_delta_gate=-0.90,
        use_cache=False,
    )
    path, report = run_single_trade_delta_hedge(md, cfg)
    # Should exercise before end in a deep ITM, low-time-value scenario (heuristic)
    assert path["option_alive"].iloc[-1] is False
