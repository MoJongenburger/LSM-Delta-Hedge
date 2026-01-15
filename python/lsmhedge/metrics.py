from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())


def max_drawdown_pct(equity: pd.Series) -> float:
    eq = equity.astype(float)
    peak = eq.cummax()
    dd_pct = (eq / peak) - 1.0
    return float(dd_pct.min())


def var_quantile(x: pd.Series, q: float) -> float:
    return float(np.nanquantile(x.astype(float).values, q))


@dataclass(frozen=True)
class HedgeReport:
    n_days: int
    final_pnl: float
    pnl_mean_daily: float
    pnl_std_daily: float
    pnl_ann_vol: float
    sharpe_ann: float
    max_dd: float
    max_dd_pct: float
    var_95_daily: float
    cvar_95_daily: float
    total_turnover: float
    total_tcost: float


def summarize_hedge(df: pd.DataFrame, value_col: str = "book_value") -> HedgeReport:
    """
    Expects df to contain:
      - book_value
      - trade (shares traded each day)
      - tcost (transaction cost paid each day)
    """
    if value_col not in df.columns:
        raise ValueError(f"Missing column: {value_col}")

    equity = df[value_col].astype(float)
    pnl = equity - float(equity.iloc[0])
    dp = pnl.diff().fillna(0.0)

    n = int(len(df))
    mu = float(dp.mean())
    sd = float(dp.std(ddof=1)) if n > 1 else 0.0
    ann_vol = float(sd * np.sqrt(252.0))
    sharpe = float((mu / sd) * np.sqrt(252.0)) if sd > 0 else 0.0

    dd = max_drawdown(equity)
    dd_pct = max_drawdown_pct(equity)

    var95 = var_quantile(dp, 0.05)
    tail = dp[dp <= var95]
    cvar95 = float(tail.mean()) if len(tail) > 0 else var95

    turnover = float(df["trade"].abs().sum()) if "trade" in df.columns else 0.0
    tcost = float(df["tcost"].sum()) if "tcost" in df.columns else 0.0

    return HedgeReport(
        n_days=n,
        final_pnl=float(pnl.iloc[-1]),
        pnl_mean_daily=mu,
        pnl_std_daily=sd,
        pnl_ann_vol=ann_vol,
        sharpe_ann=sharpe,
        max_dd=dd,
        max_dd_pct=dd_pct,
        var_95_daily=var95,
        cvar_95_daily=cvar95,
        total_turnover=turnover,
        total_tcost=tcost,
    )


def as_dict(report: HedgeReport) -> Dict[str, float]:
    return {
        "n_days": float(report.n_days),
        "final_pnl": report.final_pnl,
        "pnl_mean_daily": report.pnl_mean_daily,
        "pnl_std_daily": report.pnl_std_daily,
        "pnl_ann_vol": report.pnl_ann_vol,
        "sharpe_ann": report.sharpe_ann,
        "max_drawdown": report.max_dd,
        "max_drawdown_pct": report.max_dd_pct,
        "VaR_95_daily": report.var_95_daily,
        "CVaR_95_daily": report.cvar_95_daily,
        "total_turnover_shares": report.total_turnover,
        "total_transaction_cost": report.total_tcost,
    }

