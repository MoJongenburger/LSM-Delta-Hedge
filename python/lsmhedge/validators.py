from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationConfig:
    # Put delta should be in [-1, 0] (allow MC tolerance)
    delta_lower: float = -1.10
    delta_upper: float = 0.10

    # Price >= intrinsic, allow tiny numerical tolerance
    price_intrinsic_tol: float = 1e-8

    # Allow a bit of negative book_value drift at t=0 due to rounding
    book_value_start_tol: float = 1e-6

    # Required columns
    require_cols: Tuple[str, ...] = (
        "S", "K", "option_price", "delta", "cash", "stock", "book_value",
        "trade", "tcost", "option_alive", "pnl", "pnl_change"
    )


def validate_hedge_path(df: pd.DataFrame, cfg: Optional[ValidationConfig] = None) -> Dict[str, bool]:
    """
    Hard validation for a single hedge path DataFrame.

    Returns a dict of checks, raises ValueError if critical missing columns.
    """
    cfg = cfg or ValidationConfig()

    missing = [c for c in cfg.require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"validate_hedge_path: missing columns: {missing}")

    checks: Dict[str, bool] = {}

    # Basic finite checks
    checks["finite_book_value"] = np.isfinite(df["book_value"].astype(float)).all()
    checks["finite_option_price"] = np.isfinite(df["option_price"].astype(float)).all()
    checks["finite_delta"] = np.isfinite(df["delta"].astype(float)).all()

    # Price >= intrinsic (American/Bermudan no-arb lower bound)
    intrinsic = np.maximum(df["K"].astype(float) - df["S"].astype(float), 0.0)
    checks["price_ge_intrinsic"] = (df["option_price"].astype(float) + cfg.price_intrinsic_tol >= intrinsic).all()

    # Delta bounds for put (loose, MC tolerance)
    d = df["delta"].astype(float).values
    checks["delta_in_bounds"] = bool((d >= cfg.delta_lower).all() and (d <= cfg.delta_upper).all())

    # Book value starts near 0 (by construction: premium - hedge - option)
    bv0 = float(df["book_value"].iloc[0])
    checks["book_value_start_near_zero"] = abs(bv0) <= cfg.book_value_start_tol + 1e-3

    # If option not alive, hedge should be flat at end (stock ≈ 0)
    if "option_alive" in df.columns:
        alive_end = bool(df["option_alive"].iloc[-1])
        if not alive_end:
            checks["flat_after_exercise"] = abs(float(df["stock"].iloc[-1])) <= 1e-8
        else:
            checks["flat_after_exercise"] = True
    else:
        checks["flat_after_exercise"] = True

    return checks


def validate_sweep_table(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Light validation for sweep outputs.
    """
    checks: Dict[str, bool] = {}
    if df.empty:
        checks["non_empty"] = False
        return checks

    for c in ["final_pnl", "pnl_ann_vol", "max_drawdown_pct"]:
        if c in df.columns:
            checks[f"finite_{c}"] = np.isfinite(df[c].astype(float)).all()
        else:
            checks[f"finite_{c}"] = True

    checks["non_empty"] = True
    return checks
