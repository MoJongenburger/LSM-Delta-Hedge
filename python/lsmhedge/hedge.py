from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import price_and_delta_bermudan_put
from .metrics import summarize_hedge, as_dict


@dataclass
class HedgeConfig:
    trading_days: int = 252

    # option
    horizon_days: int = 252
    strike_mode: str = "atm"             # "atm" or "fixed"
    fixed_strike: Optional[float] = None

    # engine
    eps_rel: float = 1e-4
    engine_cfg: Optional[Dict[str, Any]] = None

    # costs
    tc_bps: float = 0.0

    # early exercise heuristic:
    # For Bermudan/American: option_value >= intrinsic always (no-arb).
    # We approximate the exercise region when time value ~ 0:
    #    option_price - payoff <= k * price_stderr
    # plus a robustness gate that delta is sufficiently close to -1 (deep ITM put).
    exercise_detection: bool = True
    exercise_k_stderr: float = 2.0
    exercise_delta_gate: float = -0.95

    # speed: optional LRU caching of pricing calls using rounded state (approximate!)
    use_cache: bool = False
    cache_maxsize: int = 512
    cache_round_S: int = 2
    cache_round_sigma: int = 4
    cache_round_r: int = 4
    cache_round_q: int = 4
    cache_round_T: int = 4

    def __post_init__(self):
        if self.engine_cfg is None:
            self.engine_cfg = {
                "steps": 252,
                "paths": 50_000,
                "train_fraction": 0.5,
                "basis_degree": 2,
                "basis": "Laguerre",
                "antithetic": True,
                "ridge": 1e-8,
                "use_control_variate": True,
                "seed": 42,  # deterministic by default
            }


class LRUQuoteCache:
    """
    LRU cache for (price, delta, price_se, delta_se) quotes.
    NOTE: This is approximate: rounding makes cache hits possible.
    Correct caching for a desk would be a reusable C++ model object (Step 6).
    """
    def __init__(self, maxsize: int = 512):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[Tuple[float, float, float, float, float], Tuple[float, float, float, float]]" = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.maxsize:
            self._d.popitem(last=False)


def _tc_cost(trade_shares: float, price: float, tc_bps: float) -> float:
    if tc_bps <= 0.0:
        return 0.0
    notional = abs(trade_shares) * price
    return float(notional * (tc_bps / 1e4))


def _cache_key(S: float, sigma: float, r: float, q: float, T: float, cfg: HedgeConfig) -> Tuple[float, float, float, float, float]:
    return (
        round(float(S), cfg.cache_round_S),
        round(float(sigma), cfg.cache_round_sigma),
        round(float(r), cfg.cache_round_r),
        round(float(q), cfg.cache_round_q),
        round(float(T), cfg.cache_round_T),
    )


def run_single_trade_delta_hedge(
    market: pd.DataFrame,
    cfg: HedgeConfig,
    start_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    market must have columns: close, sigma, r, q (DatetimeIndex).

    Book accounting:
      book_value = cash + stock*S - option_price
      cash accrues at r (continuous comp, daily dt=1/252)
      transaction cost = bps of notional traded

    At t=0:
      receive premium first, then hedge to delta.
    """
    for col in ["close", "sigma", "r", "q"]:
        if col not in market.columns:
            raise ValueError(f"market missing column: {col}")

    df = market.copy().sort_index()

    # choose start index
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        start_idx = int(df.index.searchsorted(start_ts))
        if start_idx >= len(df.index):
            raise ValueError("start_date beyond available data")
    else:
        start_idx = 0

    expiry_idx = min(start_idx + cfg.horizon_days, len(df.index) - 1)
    if expiry_idx <= start_idx:
        raise ValueError("Not enough data for hedge horizon.")

    path = df.iloc[start_idx : expiry_idx + 1].copy()
    dates = path.index
    n = len(path)

    S_init = float(path["close"].iloc[0])
    if cfg.strike_mode == "atm":
        K = S_init
    elif cfg.strike_mode == "fixed":
        if cfg.fixed_strike is None:
            raise ValueError("fixed_strike must be provided when strike_mode='fixed'")
        K = float(cfg.fixed_strike)
    else:
        raise ValueError("strike_mode must be 'atm' or 'fixed'")

    cash = np.float64(0.0)
    stock = np.float64(0.0)
    option_alive = True

    cache = LRUQuoteCache(cfg.cache_maxsize) if cfg.use_cache else None
    rows = []

    for i in range(n):
        date = dates[i]
        S = float(path["close"].iloc[i])
        sigma = float(path["sigma"].iloc[i])
        r = float(path["r"].iloc[i])
        q = float(path["q"].iloc[i])

        # cash accrual from previous day to today
        if i > 0:
            r_prev = float(path["r"].iloc[i - 1])
            dt = 1.0 / float(cfg.trading_days)
            cash = np.float64(float(cash) * float(np.exp(r_prev * dt)))

        T_rem = float((n - 1 - i) / float(cfg.trading_days))
        payoff = max(K - S, 0.0)

        trade = 0.0
        tcost = 0.0
        option_price = 0.0
        delta = 0.0
        price_se = 0.0
        delta_se = 0.0
        exercised_now = False

        if option_alive:
            if T_rem <= 0.0:
                option_price = payoff
                delta = -1.0 if S < K else 0.0
            else:
                if cache is not None:
                    key = _cache_key(S, sigma, r, q, T_rem, cfg)
                    hit = cache.get(key)
                    if hit is not None:
                        option_price, delta, price_se, delta_se = hit
                    else:
                        option_price, delta, price_se, delta_se = price_and_delta_bermudan_put(
                            S0=S, K=K, r=r, q=q, sigma=sigma, T=T_rem, eps_rel=cfg.eps_rel, **cfg.engine_cfg
                        )
                        cache.put(key, (option_price, delta, price_se, delta_se))
                else:
                    option_price, delta, price_se, delta_se = price_and_delta_bermudan_put(
                        S0=S, K=K, r=r, q=q, sigma=sigma, T=T_rem, eps_rel=cfg.eps_rel, **cfg.engine_cfg
                    )

            # Early exercise heuristic: time value approx zero + delta gate
            if cfg.exercise_detection and payoff > 0.0 and T_rem > 0.0:
                time_value = float(option_price - payoff)
                if (time_value <= float(cfg.exercise_k_stderr) * float(max(price_se, 0.0))) and (float(delta) <= float(cfg.exercise_delta_gate)):
                    exercised_now = True

            # --- t=0: receive premium first ---
            if i == 0:
                cash = np.float64(float(cash) + float(option_price))

            # rebalance hedge at close
            target_stock = np.float64(float(delta))
            trade = float(target_stock - stock)
            tcost = _tc_cost(trade, S, cfg.tc_bps)
            cash = np.float64(float(cash) - trade * S - tcost)
            stock = target_stock

            # if exercised, settle payoff and flatten hedge (include TC)
            if exercised_now:
                cash = np.float64(float(cash) - payoff)

                trade2 = float(-stock)
                tcost2 = _tc_cost(trade2, S, cfg.tc_bps)
                cash = np.float64(float(cash) - trade2 * S - tcost2)

                trade += trade2
                tcost += tcost2

                stock = np.float64(0.0)
                option_alive = False
                option_price = 0.0
                delta = 0.0
                price_se = 0.0
                delta_se = 0.0

        book_value = float(cash + stock * np.float64(S) - np.float64(option_price))

        rows.append(
            {
                "date": date,
                "S": S,
                "K": K,
                "sigma": sigma,
                "r": r,
                "q": q,
                "T_rem": T_rem,
                "payoff": payoff,
                "option_price": option_price,
                "delta": delta,
                "price_stderr": price_se,
                "delta_stderr": delta_se,
                "cash": float(cash),
                "stock": float(stock),
                "trade": float(trade),
                "tcost": float(tcost),
                "exercised_now": bool(exercised_now),
                "option_alive": bool(option_alive),
                "book_value": float(book_value),
            }
        )

        if not option_alive:
            break

    out = pd.DataFrame(rows).set_index("date")
    out["pnl"] = out["book_value"] - float(out["book_value"].iloc[0])
    out["pnl_change"] = out["pnl"].diff().fillna(0.0)

    rep = summarize_hedge(out, value_col="book_value")
    return out, as_dict(rep)
