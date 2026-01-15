from __future__ import annotations

from typing import Any, Tuple
import warnings

lsm_cpp = None
_import_error = None

# Try package-local extension first, then fallback to top-level
try:
    from . import lsm_cpp as _lsm_cpp  # type: ignore
    lsm_cpp = _lsm_cpp
except Exception as e1:
    try:
        import lsm_cpp as _lsm_cpp  # type: ignore
        lsm_cpp = _lsm_cpp
    except Exception as e2:
        _import_error = (e1, e2)


def _require_cpp() -> None:
    if lsm_cpp is None:
        raise ImportError(
            "C++ extension module not found. Build it via CMake/pybind11.\n"
            f"Original errors: {_import_error}"
        )


def price_bermudan_put(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    **cfg_kwargs: Any
) -> Tuple[float, float]:
    """
    Pythonic wrapper around the C++ LSM Bermudan put pricer.

    cfg_kwargs (optional):
        steps, paths, train_fraction, seed, basis_degree,
        basis=("Monomial"|"Laguerre"), antithetic, ridge, use_control_variate

    Returns: (price, mc_stderr)
    """
    _require_cpp()

    # Cheap, friendly validation (keep it light)
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    # Defaults live in C++ cfg; we only enforce a few practical bounds here
    train_fraction = cfg_kwargs.get("train_fraction", 0.5)
    if not (0.1 <= float(train_fraction) <= 0.9):
        raise ValueError("train_fraction must be in [0.1, 0.9]")

    paths = int(cfg_kwargs.get("paths", 100_000))
    if paths < 5_000:
        warnings.warn("paths < 5000 may cause regression instability / noisy prices.", UserWarning)

    cfg = lsm_cpp.LSMConfig()
    # Apply kwargs to config (strict keys)
    for k, v in cfg_kwargs.items():
        if not hasattr(cfg, k):
            raise TypeError(f"Unknown config key: {k}")
        # Allow basis to be passed as string
        if k == "basis" and isinstance(v, str):
            v = getattr(lsm_cpp.BasisType, v)
        setattr(cfg, k, v)

    res = lsm_cpp.price_bermudan_put_lsm(float(S0), float(K), float(r), float(q), float(sigma), float(T), cfg)
    return float(res.price), float(res.mc_stderr)
