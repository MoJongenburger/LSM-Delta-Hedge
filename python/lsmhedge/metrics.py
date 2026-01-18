from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HedgeReport:
    n_days: float
    final_pnl: float
    pnl_mean_daily: float
    pnl_std_daily: float
    pnl_ann_vol: float
    sharpe_ann: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_ref: float
    VaR_95_daily: float
    CVaR_95_daily: float
    total_turnover_shares: float
    total_transaction_cost: float


def _max_drawdown(series: np.ndarray) -> float:
    # drawdown = x - running_peak(x)
    running_peak = np.maximum.accumulate(series)
    dd = series - running_peak
    return float(np.min(dd))


def _var_cvar(x: np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
    # empirical VaR/CVaR on daily changes
    q = np.quantile(x, 1.0 - alpha)
    tail = x[x <= q]
    cvar = float(np.mean(tail)) if tail.size > 0 else float(q)
    return float(q), cvar


def summarize_hedge(df: pd.DataFrame, value_col: str = "book_value") -> HedgeReport:
    if value_col not in df.columns:
        raise ValueError(f"summarize_hedge: missing value_col={value_col}")

    v = df[value_col].astype(float).values
    n = float(len(v))

    # PnL defined as value - initial value (matches hedge.py)
    pnl = (v - float(v[0]))
    pnl_chg = np.diff(pnl, prepend=pnl[0])

    mu = float(np.mean(pnl_chg))
    sd = float(np.std(pnl_chg, ddof=1)) if len(pnl_chg) > 1 else 0.0
    ann_vol = float(sd * np.sqrt(252.0))

    # Sharpe on daily pnl changes (risk-free already accounted in cash accrual)
    sharpe = float((mu / sd) * np.sqrt(252.0)) if sd > 1e-12 else 0.0

    mdd = _max_drawdown(v)

    # Stable drawdown % reference:
    # Prefer premium0 if available, else fallback to max(1, |initial value|)
    if "premium0" in df.columns:
        ref = float(abs(float(df["premium0"].iloc[0])))
        if ref < 1e-6:
            ref = max(1.0, abs(float(v[0])))
    else:
        ref = max(1.0, abs(float(v[0])))

    mdd_pct = float(100.0 * (mdd / ref))  # percentage of reference amount

    var95, cvar95 = _var_cvar(pnl_chg, alpha=0.95)

    turnover = float(df["trade"].astype(float).abs().sum()) if "trade" in df.columns else float("nan")
    tcost = float(df["tcost"].astype(float).sum()) if "tcost" in df.columns else float("nan")

    rep = HedgeReport(
        n_days=n,
        final_pnl=float(pnl[-1]),
        pnl_mean_daily=mu,
        pnl_std_daily=sd,
        pnl_ann_vol=ann_vol,
        sharpe_ann=sharpe,
        max_drawdown=mdd,
        max_drawdown_pct=mdd_pct,
        max_drawdown_ref=ref,
        VaR_95_daily=var95,
        CVaR_95_daily=cvar95,
        total_turnover_shares=turnover,
        total_transaction_cost=tcost,
    )
    return rep


def as_dict(rep: HedgeReport) -> Dict[str, Any]:
    return {
        "n_days": rep.n_days,
        "final_pnl": rep.final_pnl,
        "pnl_mean_daily": rep.pnl_mean_daily,
        "pnl_std_daily": rep.pnl_std_daily,
        "pnl_ann_vol": rep.pnl_ann_vol,
        "sharpe_ann": rep.sharpe_ann,
        "max_drawdown": rep.max_drawdown,
        "max_drawdown_pct": rep.max_drawdown_pct,
        "max_drawdown_ref": rep.max_drawdown_ref,
        "VaR_95_daily": rep.VaR_95_daily,
        "CVaR_95_daily": rep.CVaR_95_daily,
        "total_turnover_shares": rep.total_turnover_shares,
        "total_transaction_cost": rep.total_transaction_cost,
    }
