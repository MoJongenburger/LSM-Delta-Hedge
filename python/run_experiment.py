from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# allow running from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge.data import MarketDataConfig, prepare_market_data
from lsmhedge.hedge import HedgeConfig, run_single_trade_delta_hedge
from lsmhedge.validators import validate_hedge_path, validate_sweep_table


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_and_save(df: pd.DataFrame, outdir: str) -> None:
    ensure_dir(outdir)

    # Spot vs hedge shares
    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df["S"])
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Spot (S)")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["stock"])
    ax2.set_ylabel("Hedge shares")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spot_vs_hedge.png"))
    plt.close(fig)

    # Book value & PnL
    fig, ax = plt.subplots()
    ax.plot(df.index, df["book_value"], label="Book Value")
    ax.plot(df.index, df["pnl"], label="PnL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "book_value_pnl.png"))
    plt.close(fig)

    # PnL changes histogram
    fig, ax = plt.subplots()
    ax.hist(df["pnl_change"].astype(float).values, bins=60)
    ax.set_title("Daily PnL Changes")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pnl_change_hist.png"))
    plt.close(fig)


def baseline_run(md: pd.DataFrame, outdir: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cfg = HedgeConfig(
        horizon_days=args.horizon_days,
        tc_bps=args.tc_bps,
        use_stateful_model=bool(args.stateful),
        recalibration_policy=args.recalib_policy,
        recalibration_days=args.recalib_days,
        sigma_rel_threshold=args.sigma_threshold,
        engine_cfg={
            "paths": args.paths,
            "train_fraction": 0.5,
            "basis": "Laguerre",
            "basis_degree": 2,
            "antithetic": True,
            "ridge": 1e-8,
            "use_control_variate": True,
            "seed": args.seed,
        },
        exercise_detection=True,
    )

    df_path, report = run_single_trade_delta_hedge(md, cfg, start_date=args.start_date)

    df_path.to_csv(os.path.join(outdir, "baseline_path.csv"))
    save_json(os.path.join(outdir, "baseline_report.json"), report)

    v = validate_hedge_path(df_path)
    save_json(os.path.join(outdir, "baseline_validators.json"), v)

    if args.save_plots:
        plot_and_save(df_path, os.path.join(outdir, "plots"))

    return df_path, report


def sweep(md_cfg: MarketDataConfig, outdir: str, args: argparse.Namespace) -> pd.DataFrame:
    """
    Small sweep over tc_bps and vol_window (and optionally calibration policy).
    """
    rows: List[Dict[str, float]] = []

    tc_grid = [float(x) for x in args.sweep_tc_bps.split(",")]
    vol_windows = [int(x) for x in args.sweep_vol_windows.split(",")]

    policies = [args.recalib_policy]
    if args.sweep_policies:
        policies = [p.strip() for p in args.sweep_policies.split(",")]

    for vw in vol_windows:
        md = prepare_market_data(MarketDataConfig(
            symbol=md_cfg.symbol,
            start=md_cfg.start,
            end=md_cfg.end,
            vol_window=vw,
            rf_mode=md_cfg.rf_mode,
            rf_constant=md_cfg.rf_constant,
            dividend_yield_mode=md_cfg.dividend_yield_mode,
        ))

        for tc in tc_grid:
            for pol in policies:
                cfg = HedgeConfig(
                    horizon_days=args.horizon_days,
                    tc_bps=float(tc),
                    use_stateful_model=bool(args.stateful),
                    recalibration_policy=str(pol),
                    recalibration_days=args.recalib_days,
                    sigma_rel_threshold=args.sigma_threshold,
                    engine_cfg={
                        "paths": max(10_000, int(args.paths // 2)),  # cheaper for sweep
                        "train_fraction": 0.5,
                        "basis": "Laguerre",
                        "basis_degree": 2,
                        "antithetic": True,
                        "ridge": 1e-8,
                        "use_control_variate": True,
                        "seed": args.seed,
                    },
                    exercise_detection=True,
                )

                df_path, report = run_single_trade_delta_hedge(md, cfg, start_date=args.start_date)

                rows.append({
                    "vol_window": float(vw),
                    "tc_bps": float(tc),
                    "recalib_policy": pol,
                    **report,
                })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "sweep_summary.csv"), index=False)

    v = validate_sweep_table(out)
    save_json(os.path.join(outdir, "sweep_validators.json"), v)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSM-Delta-Hedge reproducible experiment runner")

    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--start-date", default=None, help="optional start date for hedge (YYYY-MM-DD)")
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--rf", type=float, default=0.04)

    p.add_argument("--horizon-days", type=int, default=252)
    p.add_argument("--tc-bps", type=float, default=1.0)
    p.add_argument("--paths", type=int, default=50000)
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--stateful", action="store_true", help="use C++ stateful model (desk mode)")
    p.add_argument("--recalib-policy", default="weekly", choices=["daily", "weekly", "sigma_threshold"])
    p.add_argument("--recalib-days", type=int, default=5)
    p.add_argument("--sigma-threshold", type=float, default=0.15)

    p.add_argument("--save-plots", action="store_true")

    p.add_argument("--do-sweep", action="store_true")
    p.add_argument("--sweep-tc-bps", default="0,0.5,1,2,5")
    p.add_argument("--sweep-vol-windows", default="20,60,120")
    p.add_argument("--sweep-policies", default=None, help="optional comma list: daily,weekly,sigma_threshold")

    p.add_argument("--outdir", default=None, help="output directory; default results/run_<timestamp>")
    return p.parse_args()


def main():
    args = parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(REPO_ROOT, "results", f"run_{run_id}")
    ensure_dir(outdir)

    md_cfg = MarketDataConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        vol_window=args.vol_window,
        rf_mode="constant",
        rf_constant=args.rf,
        dividend_yield_mode="none",
    )

    md = prepare_market_data(md_cfg)

    # Save config
    save_json(os.path.join(outdir, "config.json"), vars(args))
    save_json(os.path.join(outdir, "market_config.json"), md_cfg.__dict__)

    # Baseline
    df_path, report = baseline_run(md, outdir, args)
    print("Baseline report:", report)

    # Sweep
    if args.do_sweep:
        sweep_df = sweep(md_cfg, outdir, args)
        print("Sweep rows:", len(sweep_df))

    print(f"Saved results to: {outdir}")


if __name__ == "__main__":
    main()
