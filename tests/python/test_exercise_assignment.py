from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from lsmhedge import price_delta_exercise_bermudan_put
from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge
from lsmhedge.validators import validate_hedge_path


def _basic_engine_cfg(paths: int = 8000):
    return {
        "steps": 25,
        "paths": int(paths),
        "train_fraction": 0.5,
        "basis_degree": 2,
        "basis": "Laguerre",
        "antithetic": True,
        "ridge": 1e-8,
        "use_control_variate": True,
        "seed": 123,
    }


def test_stateless_quote_exposes_exact_exercise_region_when_waiting_is_dominated():
    price, delta, price_se, delta_se, intrinsic, continuation, exercise_now = price_delta_exercise_bermudan_put(
        S0=50.0,
        K=100.0,
        r=0.05,
        q=0.0,
        sigma=0.0,
        T=0.25,
        eps_rel=1e-4,
        **_basic_engine_cfg(paths=6000),
    )

    assert exercise_now is True
    assert abs(price - intrinsic) < 1e-8
    assert intrinsic >= continuation - 1e-8
    assert delta == -1.0
    assert price_se == 0.0
    assert delta_se == 0.0


def test_stateful_and_stateless_quotes_are_close_at_build_time():
    lsm_cpp = importlib.import_module("lsmhedge.lsm_cpp")

    cfg = lsm_cpp.LSMConfig()
    for k, v in _basic_engine_cfg(paths=12000).items():
        if k == "basis":
            v = getattr(lsm_cpp.BasisType, v)
        setattr(cfg, k, v)
    cfg.steps = 40

    model = lsm_cpp.LSMModel(100.0, 0.02, 0.0, 0.20, 40.0 / 252.0, cfg)
    stateful = model.quote(100.0, 0, 1e-4)

    stateless = price_delta_exercise_bermudan_put(
        S0=100.0,
        K=100.0,
        r=0.02,
        q=0.0,
        sigma=0.20,
        T=40.0 / 252.0,
        eps_rel=1e-4,
        steps=40,
        paths=12000,
        train_fraction=0.5,
        basis_degree=2,
        basis="Laguerre",
        antithetic=True,
        ridge=1e-8,
        use_control_variate=True,
        seed=123,
    )

    assert abs(float(stateful.price) - stateless[0]) < 5.0 * max(float(stateful.price_stderr), stateless[2], 1e-4)
    assert abs(float(stateful.delta) - stateless[1]) < 0.10
    assert bool(stateful.exercise_now) == bool(stateless[6])


def test_physical_assignment_flattens_book_on_termination():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    market = pd.DataFrame(
        {
            "close": [80.0, 80.0],
            "sigma": [0.20, 0.20],
            "r": [0.01, 0.01],
            "q": [0.0, 0.0],
        },
        index=dates,
    )

    cfg = HedgeConfig(
        horizon_days=1,
        strike_mode="fixed",
        fixed_strike=100.0,
        use_stateful_model=False,
        exercise_policy="none",           # hold until expiry for deterministic test
        settlement_style="physical_put",
        flatten_after_termination=True,
        tc_bps=0.0,
        engine_cfg={
            "steps": 1,
            "paths": 8000,
            "train_fraction": 0.5,
            "basis_degree": 2,
            "basis": "Laguerre",
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": 7,
        },
    )

    path, report = run_single_trade_delta_hedge(market, cfg)

    assert bool(path["option_alive"].iloc[-1]) is False
    assert bool(path["exercised_now"].iloc[-1]) is True
    assert bool(path["assigned_physical"].iloc[-1]) is True
    assert abs(float(path["assignment_shares"].iloc[-1]) - 1.0) < 1e-12
    assert abs(float(path["stock"].iloc[-1])) < 1e-10
    assert abs(float(path["option_price"].iloc[-1])) < 1e-10

    checks = validate_hedge_path(path)
    assert all(checks.values()), checks
    assert report["assigned_physical"] == 1.0
