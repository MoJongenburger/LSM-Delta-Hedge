from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationConfig:
    delta_lower: float = -1.10
    delta_upper: float = 0.10
    price_intrinsic_tol: float = 1e-8
    book_value_start_tol: float = 1e-6

    require_cols: Tuple[str, ...] = (
        "S", "K", "option_price", "delta", "cash", "stock", "book_value",
        "trade", "tcost", "option_alive", "pnl", "pnl_change",
        "model_intrinsic", "model_continuation", "model_exercise_now",
        "option_terminated", "exercised_now"
    )


def validate_hedge_path(df: pd.DataFrame, cfg: Optional[ValidationConfig] = None) -> Dict[str, bool]:
    """
    Hard validation for a single hedge path DataFrame.

    Important:
    - Price >= intrinsic is enforced only while the option is alive.
    - After termination, option_price is expected to be 0.
    """
    cfg = cfg or ValidationConfig()

    missing = [c for c in cfg.require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"validate_hedge_path: missing columns: {missing}")

    checks: Dict[str, bool] = {}

    checks["finite_book_value"] = bool(np.isfinite(df["book_value"].astype(float)).all())
    checks["finite_option_price"] = bool(np.isfinite(df["option_price"].astype(float)).all())
    checks["finite_delta"] = bool(np.isfinite(df["delta"].astype(float)).all())
    checks["finite_model_intrinsic"] = bool(np.isfinite(df["model_intrinsic"].astype(float)).all())
    checks["finite_model_continuation"] = bool(np.isfinite(df["model_continuation"].astype(float)).all())

    alive_mask = df["option_alive"].astype(bool).values
    alive_any = bool(alive_mask.any())

    intrinsic = np.maximum(df["K"].astype(float).values - df["S"].astype(float).values, 0.0)
    if alive_any:
        op_alive = df.loc[df["option_alive"].astype(bool), "option_price"].astype(float).values
        intr_alive = intrinsic[df["option_alive"].astype(bool).values]
        checks["price_ge_intrinsic"] = bool((op_alive + cfg.price_intrinsic_tol >= intr_alive).all())
    else:
        checks["price_ge_intrinsic"] = True

    if alive_any:
        d_alive = df.loc[df["option_alive"].astype(bool), "delta"].astype(float).values
        checks["delta_in_bounds"] = bool((d_alive >= cfg.delta_lower).all() and (d_alive <= cfg.delta_upper).all())
    else:
        checks["delta_in_bounds"] = True

    bv0 = float(df["book_value"].iloc[0])
    checks["book_value_start_near_zero"] = bool(abs(bv0) <= cfg.book_value_start_tol + 1e-3)

    # If the model says exercise now, intrinsic should be at least as large as continuation.
    me = df["model_exercise_now"].astype(bool).values
    if me.any():
        lhs = df.loc[df["model_exercise_now"].astype(bool), "model_intrinsic"].astype(float).values
        rhs = df.loc[df["model_exercise_now"].astype(bool), "model_continuation"].astype(float).values
        checks["exercise_policy_consistent"] = bool((lhs + 1e-8 >= rhs).all())
    else:
        checks["exercise_policy_consistent"] = True

    terminated_mask = df["option_terminated"].astype(bool).values
    if terminated_mask.any():
        op_term = df.loc[df["option_terminated"].astype(bool), "option_price"].astype(float).values
        checks["option_zero_after_termination"] = bool(np.allclose(op_term, 0.0, atol=1e-10))
    else:
        checks["option_zero_after_termination"] = True

    # Last row should have no live option if the horizon reached expiry or exercise happened.
    checks["terminal_option_inactive_if_terminated"] = bool(
        (not bool(df["option_terminated"].iloc[-1])) or (not bool(df["option_alive"].iloc[-1]))
    )

    return checks


def validate_sweep_table(df: pd.DataFrame) -> Dict[str, bool]:
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
