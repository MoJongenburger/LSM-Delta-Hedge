from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge.data import MarketDataConfig, prepare_market_data
from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge
from lsmhedge.validators import validate_hedge_path


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_start_dates(index: pd.DatetimeIndex, step_bd: int, min_horizon: int) -> List[pd.Timestamp]:
    starts: List[pd.Timestamp] = []
    for i in range(0, len(index), step_bd):
        if i + min_horizon < len(index):
            starts.append(index[i])
    return starts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-start regime study for LSM delta-hedging")

    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--vol-window", type=int, default=60)

    p.add_argument("--horizon-days", type=int, default=252)
    p.add_argument("--step-bd", type=int, default=21, help="start date spacing in business days (21 ~ monthly)")

    p.add_argument("--rf", type=float, default=0.04)
    p.add_argument("--tc-bps", type=float, default=1.0)

    p.add_argument("--paths", type=int, default=25000)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--stateful", action="store_true")
    p.add_argument("--recalib-policy", default="weekly", choices=["daily", "weekly", "sigma_threshold"])
    p.add_argument("--recalib-days", type=int, default=5)

    p.add_argument("--outdir", default=None, help="default results/regime_<timestamp>")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(REPO_ROOT, "results", f"regime_{run_id}")
    _ensure_dir(outdir)

    md = prepare_market_data(MarketDataConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        vol_window=args.vol_window,
        rf_mode="constant",
        rf_constant=args.rf,
        dividend_yield_mode="none",
    ))

    hedge_cfg = HedgeConfig(
        horizon_days=args.horizon_days,
        tc_bps=args.tc_bps,
        use_stateful_model=bool(args.stateful),
        recalibration_policy=args.recalib_policy,
        recalibration_days=args.recalib_days,
        sigma_rel_threshold=0.15,
        engine_cfg={
            "paths": int(args.paths),
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "basis_degree": 2,
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": int(args.seed),
        },
        exercise_detection=True,
    )

    starts = pick_start_dates(md.index, step_bd=int(args.step_bd), min_horizon=int(args.horizon_days))

    rows: List[Dict[str, float]] = []
    for s in starts:
        path, report = run_single_trade_delta_hedge(md, hedge_cfg, start_date=str(s.date()))

        # exercise metadata
        exercised_any = bool(("exercised_now" in path.columns) and (path["exercised_now"].astype(bool).any()))
        if exercised_any:
            ex_idx = int(np.argmax(path["exercised_now"].astype(bool).values))  # first True
            exercise_day = ex_idx
        else:
            exercise_day = -1

        # exercised early if we stopped before full horizon
        exercised_early = exercised_any and (len(path) < (args.horizon_days + 1))

        pct_completed = float(len(path) / float(args.horizon_days + 1))

        v = validate_hedge_path(path)
        v_pass = float(all(v.values()))

        rows.append({
            "start_date": str(s.date()),
            "end_date": str(path.index[-1].date()),
            "n_days": float(report.get("n_days", len(path))),
            "pct_horizon_completed": pct_completed,
            "exercised_early": float(exercised_early),
            "exercise_day": float(exercise_day),
            "final_pnl": float(report["final_pnl"]),
            "pnl_ann_vol": float(report["pnl_ann_vol"]),
            "sharpe_ann": float(report.get("sharpe_ann", np.nan)),
            "max_drawdown": float(report["max_drawdown"]),
            "max_drawdown_pct": float(report["max_drawdown_pct"]),
            "max_drawdown_ref": float(report.get("max_drawdown_ref", np.nan)),
            "VaR_95_daily": float(report["VaR_95_daily"]),
            "CVaR_95_daily": float(report["CVaR_95_daily"]),
            "total_turnover_shares": float(report.get("total_turnover_shares", np.nan)),
            "total_transaction_cost": float(report.get("total_transaction_cost", np.nan)),
            "validators_pass": v_pass,
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "regime_summary.csv"), index=False)

    meta = {
        "symbol": args.symbol,
        "start": args.start,
        "end": args.end,
        "vol_window": args.vol_window,
        "horizon_days": args.horizon_days,
        "step_bd": args.step_bd,
        "rf": args.rf,
        "tc_bps": args.tc_bps,
        "paths": args.paths,
        "seed": args.seed,
        "stateful": bool(args.stateful),
        "recalibration_policy": args.recalib_policy,
        "recalibration_days": args.recalib_days,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(outdir, "regime_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(out.describe(include="all"))
    print(f"Saved: {outdir}")


if __name__ == "__main__":
    main()
