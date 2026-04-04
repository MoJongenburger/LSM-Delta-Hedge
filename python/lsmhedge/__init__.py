from __future__ import annotations

from typing import Any, Tuple
import warnings
import importlib

_lsm_cpp = None


def _load_cpp():
    """
    Load the compiled extension in a deterministic way.

    Priority:
      1) package-local extension: lsmhedge.lsm_cpp
      2) top-level module: lsm_cpp
    """
    global _lsm_cpp
    if _lsm_cpp is not None:
        return _lsm_cpp

    errors = []

    try:
        _lsm_cpp = importlib.import_module("lsmhedge.lsm_cpp")
        return _lsm_cpp
    except Exception as e:
        errors.append(("lsmhedge.lsm_cpp", repr(e)))

    try:
        _lsm_cpp = importlib.import_module("lsm_cpp")
        return _lsm_cpp
    except Exception as e:
        errors.append(("lsm_cpp", repr(e)))

    raise ImportError(
        "C++ extension module not found. Build it via CMake/pybind11.\n"
        f"Import attempts failed with: {errors}"
    )


def _build_cfg(**cfg_kwargs: Any):
    lsm_cpp = _load_cpp()
    cfg = lsm_cpp.LSMConfig()
    for k, v in cfg_kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown config key: {k}")
        if k == "basis" and isinstance(v, str):
            v = getattr(lsm_cpp.BasisType, v)
        setattr(cfg, k, v)
    return cfg


def price_bermudan_put(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    **cfg_kwargs: Any
) -> Tuple[float, float]:
    lsm_cpp = _load_cpp()

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    train_fraction = cfg_kwargs.get("train_fraction", 0.5)
    if not (0.1 <= float(train_fraction) <= 0.9):
        raise ValueError("train_fraction must be in [0.1, 0.9]")

    paths = int(cfg_kwargs.get("paths", 100_000))
    if paths < 5_000:
        warnings.warn("paths < 5000 may cause regression instability / noisy prices.", UserWarning)

    cfg = _build_cfg(**cfg_kwargs)
    res = lsm_cpp.price_bermudan_put_lsm(float(S0), float(K), float(r), float(q), float(sigma), float(T), cfg)
    return float(res.price), float(res.mc_stderr)


def price_and_delta_bermudan_put(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    eps_rel: float = 1e-4,
    **cfg_kwargs: Any
) -> Tuple[float, float, float, float]:
    """
    Backward-compatible public wrapper: returns only price/delta/stderrs.
    """
    out = price_delta_exercise_bermudan_put(
        S0=S0, K=K, r=r, q=q, sigma=sigma, T=T, eps_rel=eps_rel, **cfg_kwargs
    )
    return out[0], out[1], out[2], out[3]


def price_delta_exercise_bermudan_put(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    eps_rel: float = 1e-4,
    **cfg_kwargs: Any
) -> Tuple[float, float, float, float, float, float, bool]:
    """
    Rich wrapper used by the hedge engine.

    Returns:
        price, delta, price_stderr, delta_stderr,
        intrinsic_value, continuation_value, exercise_now
    """
    lsm_cpp = _load_cpp()

    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if not (1e-6 <= float(eps_rel) <= 1e-2):
        raise ValueError("eps_rel must be in [1e-6, 1e-2]")

    cfg = _build_cfg(**cfg_kwargs)
    res = lsm_cpp.price_and_delta_bermudan_put_lsm(
        float(S0), float(K), float(r), float(q), float(sigma), float(T), float(eps_rel), cfg
    )
    return (
        float(res.price),
        float(res.delta),
        float(res.price_stderr),
        float(res.delta_stderr),
        float(res.intrinsic_value),
        float(res.continuation_value),
        bool(res.exercise_now),
    )
