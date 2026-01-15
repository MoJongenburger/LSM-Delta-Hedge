from __future__ import annotations

try:
    import lsm_cpp  # built by CMake/pybind11
except ImportError as e:
    lsm_cpp = None
    _import_error = e


def _require_cpp():
    if lsm_cpp is None:
        raise ImportError(
            "lsm_cpp module not found. Build it with CMake first (pybind11). "
            f"Original error: {_import_error}"
        )


def price_bermudan_put(S0: float, K: float, r: float, q: float, sigma: float, T: float,
                       steps: int = 252, paths: int = 100_000, train_fraction: float = 0.5,
                       basis_degree: int = 2, basis: str = "Laguerre",
                       seed: int = 42, antithetic: bool = True,
                       ridge: float = 1e-8, use_control_variate: bool = True):
    """
    Convenience wrapper around lsm_cpp.price_bermudan_put_lsm.
    Returns (price, mc_stderr).
    """
    _require_cpp()

    cfg = lsm_cpp.LSMConfig()
    cfg.steps = int(steps)
    cfg.paths = int(paths)
    cfg.train_fraction = float(train_fraction)
    cfg.seed = int(seed)
    cfg.basis_degree = int(basis_degree)
    cfg.basis = getattr(lsm_cpp.BasisType, basis)
    cfg.antithetic = bool(antithetic)
    cfg.ridge = float(ridge)
    cfg.use_control_variate = bool(use_control_variate)

    res = lsm_cpp.price_bermudan_put_lsm(float(S0), float(K), float(r), float(q), float(sigma), float(T), cfg)
    return res.price, res.mc_stderr

