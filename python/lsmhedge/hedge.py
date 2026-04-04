from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import price_delta_exercise_bermudan_put
from .metrics import summarize_hedge, as_dict
from .model_cache import LSMModelManager, RecalibrationConfig


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

    # exercise handling
    exercise_policy: str = "model"       # "model" | "heuristic" | "none"
    settlement_style: str = "physical_put"  # "physical_put" | "cash"
    flatten_after_termination: bool = True

    # legacy heuristic (optional)
    exercise_detection: bool = True
    exercise_k_stderr: float = 2.0
    exercise_delta_gate: float = -0.95

    # desk-dev
    use_stateful_model: bool = True
    recalibration_policy: str = "weekly"     # "daily" | "weekly" | "sigma_threshold"
    recalibration_days: int = 5
    sigma_rel_threshold: float = 0.15
    r_abs_threshold: float = 0.005
    q_abs_threshold: float = 0.005

    # legacy cache
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
                "seed": 42,
            }


class LRUQuoteCache:
    def __init__(self, maxsize: int = 512):
        self.maxsize = int(maxsize)
        self._d: "OrderedDict[Tuple[float, float, float, float, float], Tuple[float, float, float, float, float, float, bool]]" = OrderedDict()

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


def _should_terminate_early(
    cfg: HedgeConfig,
    *,
    payoff: float,
    option_price: float,
    delta: float,
    price_se: float,
    model_exercise_now: bool,
) -> bool:
    policy = str(cfg.exercise_policy).lower()

    if policy == "none":
        return False
    if policy == "model":
        return bool(model_exercise_now)
    if policy == "heuristic":
        if not cfg.exercise_detection or payoff <= 0.0:
            return False
        time_value = float(option_price - payoff)
        return (
            time_value <= float(cfg.exercise_k_stderr) * float(max(price_se, 0.0))
            and float(delta) <= float(cfg.exercise_delta_gate)
        )

    raise ValueError("exercise_policy must be one of: 'model', 'heuristic', 'none'")


def run_single_trade_delta_hedge(
    market: pd.DataFrame,
    cfg: HedgeConfig,
    start_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    for col in ["close", "sigma", "r", "q"]:
        if col not in market.columns:
            raise ValueError(f"market missing column: {col}")

    df = market.copy().sort_index()

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

    path = df.iloc[start_idx: expiry_idx + 1].copy()
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

    premium0 = np.nan

    manager = None
    if cfg.use_stateful_model:
        rcfg = RecalibrationConfig(
            policy=cfg.recalibration_policy,
            recalibration_days=cfg.recalibration_days,
            sigma_rel_threshold=cfg.sigma_rel_threshold,
            r_abs_threshold=cfg.r_abs_threshold,
            q_abs_threshold=cfg.q_abs_threshold,
        )
        manager = LSMModelManager(cfg.engine_cfg, trading_days=cfg.trading_days, rcfg=rcfg)

    cache = LRUQuoteCache(cfg.cache_maxsize) if (cfg.use_cache and not cfg.use_stateful_model) else None

    rows = []

    for i in range(n):
        date = dates[i]
        S = float(path["close"].iloc[i])
        sigma = float(path["sigma"].iloc[i])
        r = float(path["r"].iloc[i])
        q = float(path["q"].iloc[i])

        if i > 0:
            r_prev = float(path["r"].iloc[i - 1])
            dt = 1.0 / float(cfg.trading_days)
            cash = np.float64(float(cash) * float(np.exp(r_prev * dt)))

        remaining_days = int(n - 1 - i)
        T_rem = float(remaining_days / float(cfg.trading_days))
        payoff = max(K - S, 0.0)

        trade = 0.0
        tcost = 0.0
        option_price = 0.0
        delta = 0.0
        price_se = 0.0
        delta_se = 0.0
        model_intrinsic = payoff
        model_continuation = 0.0
        model_exercise_now = False
        exercised_now = False
        assigned_physical = False
        assignment_shares = 0.0
        rebuilt_model = False
        model_age = None
        option_terminated = False

        if option_alive:
            if remaining_days <= 0:
                option_price = payoff
                delta = -1.0 if S < K else 0.0
                model_intrinsic = payoff
                model_continuation = 0.0
                model_exercise_now = bool(payoff > 0.0)
            else:
                if cfg.use_stateful_model and manager is not None:
                    (
                        option_price,
                        delta,
                        price_se,
                        delta_se,
                        model_intrinsic,
                        model_continuation,
                        model_exercise_now,
                        rebuilt_model,
                        model_age,
                    ) = manager.quote(
                        day_idx=i,
                        remaining_days=remaining_days,
                        S=S,
                        K=K,
                        r=r,
                        q=q,
                        sigma=sigma,
                        eps_rel=cfg.eps_rel,
                    )
                else:
                    if cache is not None:
                        key = _cache_key(S, sigma, r, q, T_rem, cfg)
                        hit = cache.get(key)
                        if hit is not None:
                            (
                                option_price,
                                delta,
                                price_se,
                                delta_se,
                                model_intrinsic,
                                model_continuation,
                                model_exercise_now,
                            ) = hit
                        else:
                            quote = price_delta_exercise_bermudan_put(
                                S0=S, K=K, r=r, q=q, sigma=sigma, T=T_rem, eps_rel=cfg.eps_rel, **cfg.engine_cfg
                            )
                            cache.put(key, quote)
                            (
                                option_price,
                                delta,
                                price_se,
                                delta_se,
                                model_intrinsic,
                                model_continuation,
                                model_exercise_now,
                            ) = quote
                    else:
                        (
                            option_price,
                            delta,
                            price_se,
                            delta_se,
                            model_intrinsic,
                            model_continuation,
                            model_exercise_now,
                        ) = price_delta_exercise_bermudan_put(
                            S0=S, K=K, r=r, q=q, sigma=sigma, T=T_rem, eps_rel=cfg.eps_rel, **cfg.engine_cfg
                        )

            if i == 0:
                premium0 = float(option_price)
                cash = np.float64(float(cash) + float(option_price))

            terminate_for_expiry = bool(remaining_days <= 0)
            terminate_early = _should_terminate_early(
                cfg,
                payoff=payoff,
                option_price=option_price,
                delta=delta,
                price_se=price_se,
                model_exercise_now=model_exercise_now,
            )
            option_terminated = bool(terminate_for_expiry or terminate_early)

            if option_terminated:
                exercised_now = bool(payoff > 0.0 and (terminate_for_expiry or terminate_early))

                if exercised_now:
                    if cfg.settlement_style == "physical_put":
                        cash = np.float64(float(cash) - float(K))
                        stock = np.float64(float(stock) + 1.0)
                        assigned_physical = True
                        assignment_shares = 1.0
                    elif cfg.settlement_style == "cash":
                        cash = np.float64(float(cash) - float(payoff))
                    else:
                        raise ValueError("settlement_style must be 'physical_put' or 'cash'")

                option_alive = False
                option_price = 0.0
                delta = 0.0
                price_se = 0.0
                delta_se = 0.0

                if cfg.flatten_after_termination:
                    target_stock = np.float64(0.0)
                    trade = float(target_stock - stock)
                    tcost = _tc_cost(trade, S, cfg.tc_bps)
                    cash = np.float64(float(cash) - trade * S - tcost)
                    stock = target_stock
            else:
                target_stock = np.float64(float(delta))
                trade = float(target_stock - stock)
                tcost = _tc_cost(trade, S, cfg.tc_bps)
                cash = np.float64(float(cash) - trade * S - tcost)
                stock = target_stock

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
                "model_intrinsic": float(model_intrinsic),
                "model_continuation": float(model_continuation),
                "model_exercise_now": bool(model_exercise_now),
                "cash": float(cash),
                "stock": float(stock),
                "trade": float(trade),
                "tcost": float(tcost),
                "exercised_now": bool(exercised_now),
                "assigned_physical": bool(assigned_physical),
                "assignment_shares": float(assignment_shares),
                "option_terminated": bool(option_terminated),
                "option_alive": bool(option_alive),
                "book_value": float(book_value),
                "model_rebuilt": bool(rebuilt_model),
                "model_age_steps": (int(model_age) if model_age is not None else None),
                "premium0": float(premium0) if np.isfinite(premium0) else np.nan,
            }
        )

        if not option_alive:
            break

    out = pd.DataFrame(rows).set_index("date")
    out["pnl"] = out["book_value"] - float(out["book_value"].iloc[0])
    out["pnl_change"] = out["pnl"].diff().fillna(0.0)

    rep = summarize_hedge(out, value_col="book_value")
    report = as_dict(rep)
    report["exercised_early"] = float(bool("exercised_now" in out.columns and out["exercised_now"].iloc[:-1].astype(bool).any()))
    report["assigned_physical"] = float(bool("assigned_physical" in out.columns and out["assigned_physical"].astype(bool).any()))
    return out, report
