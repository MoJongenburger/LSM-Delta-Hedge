from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib
import copy


def _cpp():
    # Prefer package-local extension
    try:
        return importlib.import_module("lsmhedge.lsm_cpp")
    except Exception:
        return importlib.import_module("lsm_cpp")


@dataclass
class RecalibrationConfig:
    policy: str = "weekly"           # "daily" | "weekly" | "sigma_threshold"
    recalibration_days: int = 5
    sigma_rel_threshold: float = 0.15
    r_abs_threshold: float = 0.005
    q_abs_threshold: float = 0.005


class LSMModelManager:
    """
    Holds one active C++ LSMModel and rebuilds it according to a calibration policy.
    After build at day t0, quote with start_step = (t - t0).
    """
    def __init__(self, engine_cfg: Dict[str, Any], trading_days: int = 252, rcfg: Optional[RecalibrationConfig] = None):
        self.engine_cfg = dict(engine_cfg)
        self.trading_days = int(trading_days)
        self.rcfg = rcfg or RecalibrationConfig()

        self._m = _cpp()
        self.model = None
        self.model_day0: Optional[int] = None

        self.ref_sigma: Optional[float] = None
        self.ref_r: Optional[float] = None
        self.ref_q: Optional[float] = None

    def _make_cfg(self, steps: int):
        cfg = self._m.LSMConfig()
        # Apply defaults from engine_cfg
        for k, v in self.engine_cfg.items():
            if hasattr(cfg, k):
                if k == "basis" and isinstance(v, str):
                    v = getattr(self._m.BasisType, v)
                setattr(cfg, k, v)
        cfg.steps = int(steps)
        return cfg

    def _need_rebuild(self, day_idx: int, start_step: int, sigma: float, r: float, q: float) -> bool:
        if self.model is None or self.model_day0 is None:
            return True
        if start_step >= int(self.model.steps):
            return True

        pol = self.rcfg.policy.lower()
        if pol == "daily":
            return True
        if pol == "weekly":
            return start_step >= int(self.rcfg.recalibration_days)
        if pol == "sigma_threshold":
            if self.ref_sigma is None:
                return True
            if self.ref_sigma > 0 and abs(sigma - self.ref_sigma) / self.ref_sigma > float(self.rcfg.sigma_rel_threshold):
                return True
            if self.ref_r is not None and abs(r - self.ref_r) > float(self.rcfg.r_abs_threshold):
                return True
            if self.ref_q is not None and abs(q - self.ref_q) > float(self.rcfg.q_abs_threshold):
                return True
            return False

        # unknown policy -> rebuild
        return True

    def quote(
        self,
        *,
        day_idx: int,
        remaining_days: int,
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        eps_rel: float = 1e-4,
    ):
        """
        Returns: (price, delta, price_stderr, delta_stderr, rebuilt, start_step)
        """
        remaining_days = int(max(1, remaining_days))
        T = float(remaining_days / self.trading_days)

        if self.model is None or self.model_day0 is None:
            start_step = 0
        else:
            start_step = int(day_idx - self.model_day0)

        if self._need_rebuild(day_idx, start_step, sigma, r, q):
            cfg = self._make_cfg(steps=remaining_days)
            self.model = self._m.LSMModel(float(K), float(r), float(q), float(sigma), float(T), cfg)
            self.model_day0 = int(day_idx)

            self.ref_sigma = float(sigma)
            self.ref_r = float(r)
            self.ref_q = float(q)

            start_step = 0
            rebuilt = True
        else:
            rebuilt = False

        res = self.model.quote(float(S), int(start_step), float(eps_rel))
        return float(res.price), float(res.delta), float(res.price_stderr), float(res.delta_stderr), rebuilt, start_step
