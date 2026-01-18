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

    # Book value should start near 0-ish (depends on rounding + premium/hedge TC)
    book_value_start_tol: float = 1e-6

    require_cols: Tuple[str, ...] = (
        "S", "K", "option_price", "delta", "cash", "stock", "book_value",
        "trade", "tcost", "option_alive", "pnl", "pnl_change"
    )


def validate_hedge_path(df: pd.DataFrame, cfg: Optional[ValidationConfig] = None) -> Dict[str, bool]:
    """
    Hard validation for a single hedge path DataFrame.

    Important: Checks involving option pricing (e.g., price >= intrinsic) are enforced
    ONLY while the option is alive. After exercise, option_price is set to 0.0 by design.
    """
    cfg = cfg or ValidationConfig()

    missing = [c for c in cfg.require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"validate_hedge_path: missing columns: {missing}")

    checks: Dict[str, bool] = {}

    # Basic finite checks (cast to Python bool for JSON serialization)
    checks["finite_book_value"] = bool(np.isfinite(df["book_value"].astype(float)).all())
    checks["finite_option_price"] = bool(np.isfinite(df["option_price"].astype(float)).all())
    checks["finite_delta"] = bool(np.isfinite(df["delta"].astype(float)).all())

    # Determine alive mask
    alive = df["option_alive"].astype(bool).values
    alive_any = bool(alive.any())

    # Price >= intrinsic ONLY while alive
    intrinsic = np.maximum(df["K"].astype(float).values - df["S"].astype(float).values, 0.0)
    if alive_any:
        op_alive = df.loc[df["option_alive"].astype(bool), "option_price"].astype(float).values
        intr_alive = intrinsic[df["option_alive"].astype(bool).values]
        checks["price_ge_intrinsic"] = bool((op_alive + cfg.price_intrinsic_tol >= intr_alive).all())
    else:
        checks["price_ge_intrinsic"] = True

    # Delta bounds: enforce ONLY while alive (after exercise delta is set to 0 by design)
    if alive_any:
        d_alive = df.loc[df["option_alive"].astype(bool), "delta"].astype(float).values
        checks["delta_in_bounds"] = bool((d_alive >= cfg.delta_lower).all() and (d_alive <= cfg.delta_upper).all())
    else:
        checks["delta_in_bounds"] = True

    # Book value starts near zero-ish
    bv0 = float(df["book_value"].iloc[0])
    checks["book_value_start_near_zero"] = bool(abs(bv0) <= cfg.book_value_start_tol + 1e-3)

    # If option not alive at end, hedge should be flat
    if not bool(df["option_alive"].iloc[-1]):
        checks["flat_after_exercise"] = bool(abs(float(df["stock"].iloc[-1])) <= 1e-8)
    else:
        checks["flat_after_exercise"] = True

    return checks


def validate_sweep_table(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Light validation for sweep outputs (cast to Python bool).
    """
    checks: Dict[str, bool] = {}
    if df.empty:
        checks["non_empty"] = False
        return checks

    for c in ["final_pnl", "pnl_ann_vol", "max_drawdown_pct"]:
        if c in df.columns:
            checks[f"finite_{c}"] = bool(np.isfinite(df[c].astype(float)).all())
        else:
            checks[f"finite_{c}"] = True

    checks["non_empty"] = True
    return checks
