from __future__ import annotations

from typing import Any, Tuple
import warnings
import sys
from pathlib import Path

lsm_cpp = None
_import_error = []  # always a list of exceptions / messages


def _try_import_cpp() -> None:
    global lsm_cpp, _import_error
    errors = []

    # 1) package-local extension (preferred)
    try:
        from . import lsm_cpp as _m  # type: ignore
        lsm_cpp = _m
        _import_error = []
        return
    except Exception as e:
        errors.append(("package-local", repr(e)))

    # 2) top-level import (if module on sys.path)
    try:
        import lsm_cpp as _m  # type: ignore
        lsm_cpp = _m
        _import_error = []
        return
    except Exception as e:
        errors.append(("top-level", repr(e)))

    # 3) add repo_root/build to sys.path and try again
    try:
        repo_root = Path(__file__).resolve().parents[2]
        build_dir = repo_root / "build"
        if build_dir.exists():
            sys.path.insert(0, str(build_dir))
        import lsm_cpp as _m  # type: ignore
        lsm_cpp = _m
        _import_error = []
        return
    except Exception as e:
        errors.append(("repo-build", repr(e)))

    _import_error = errors


_try_import_cpp()


def _require_cpp() -> None:
    if lsm_cpp is None:
        raise ImportError(
            "C++ extension module not found. Build it via CMake/pybind11.\n"
            f"Import attempts failed with: {_import_error}"
        )


def _build_cfg(**cfg_kwargs: Any):
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
    _require_cpp()

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
    _require_cpp()

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
    return float(res.price), float(res.delta), float(res.price_stderr), float(res.delta_stderr)
