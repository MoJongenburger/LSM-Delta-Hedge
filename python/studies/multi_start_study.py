from __future__ import annotations

import os
import sys
import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge.data import MarketDataConfig, prepare_market_data
from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge
from lsmhedge.validators import validate_hedge_path


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_start_dates(index: pd.DatetimeIndex, step_bd: int = 21, min_horizon: int = 252) -> List[pd.Timestamp]:
    """
    Choose start dates roughly monthly (every step_bd business days),
    ensuring we have enough forward data for horizon.
    """
    starts = []
    for i in range(0, len(index), step_bd):
        if i + min_horizon < len(index):
            starts.append(index[i])
    return starts


def run_multi_start(
    outdir: str,
    symbol: str = "SPY",
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    vol_window: int = 60,
    horizon_days: int = 252,
    step_bd: int = 21,
    tc_bps: float = 1.0,
    use_stateful_model: bool = True,
    recalibration_policy: str = "weekly",
) -> pd.DataFrame:
    _ensure_dir(outdir)

    md = prepare_market_data(MarketDataConfig(
        symbol=symbol, start=start, end=end, vol_window=vol_window,
        rf_mode="constant", rf_constant=0.04, dividend_yield_mode="none"
    ))

    hedge_cfg = HedgeConfig(
        horizon_days=horizon_days,
        tc_bps=tc_bps,
        use_stateful_model=use_stateful_model,
        recalibration_policy=recalibration_policy,
        recalibration_days=5,
        sigma_rel_threshold=0.15,
        engine_cfg={
            "paths": 25_000,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "basis_degree": 2,
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": 123,
        },
        exercise_detection=True,
    )

    starts = pick_start_dates(md.index, step_bd=step_bd, min_horizon=horizon_days)

    rows: List[Dict[str, float]] = []
    for s in starts:
        path, report = run_single_trade_delta_hedge(md, hedge_cfg, start_date=str(s.date()))

        # Validators (do not raise; record pass rates)
        v = validate_hedge_path(path)
        v_pass = all(v.values())

        rows.append({
            "start_date": str(s.date()),
            "end_date": str(path.index[-1].date()),
            "n_days": float(report.get("n_days", len(path))),
            "final_pnl": float(report["final_pnl"]),
            "pnl_ann_vol": float(report["pnl_ann_vol"]),
            "sharpe_ann": float(report.get("sharpe_ann", np.nan)),
            "max_drawdown": float(report["max_drawdown"]),
            "max_drawdown_pct": float(report["max_drawdown_pct"]),
            "VaR_95_daily": float(report["VaR_95_daily"]),
            "CVaR_95_daily": float(report["CVaR_95_daily"]),
            "total_turnover_shares": float(report.get("total_turnover_shares", np.nan)),
            "total_transaction_cost": float(report.get("total_transaction_cost", np.nan)),
            "validators_pass": float(v_pass),
        })

    out = pd.DataFrame(rows)

    out_path = os.path.join(outdir, "regime_summary.csv")
    out.to_csv(out_path, index=False)

    meta = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "vol_window": vol_window,
        "horizon_days": horizon_days,
        "step_bd": step_bd,
        "tc_bps": tc_bps,
        "use_stateful_model": use_stateful_model,
        "recalibration_policy": recalibration_policy,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(outdir, "regime_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out


if __name__ == "__main__":
    # default run writes to results/regime_<timestamp>
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(REPO_ROOT, "results", f"regime_{run_id}")
    df = run_multi_start(outdir=outdir)
    print(df.describe(include="all"))
    print(f"Saved: {outdir}")
