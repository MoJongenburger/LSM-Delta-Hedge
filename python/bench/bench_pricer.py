from __future__ import annotations

import os, sys, time
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge import price_and_delta_bermudan_put


def main():
    S0, K, r, q, sigma, T = 100.0, 100.0, 0.04, 0.0, 0.2, 1.0

    cfg = dict(
        steps=252,
        train_fraction=0.5,
        basis="Laguerre",
        basis_degree=2,
        antithetic=True,
        ridge=1e-8,
        use_control_variate=True,
        seed=123,
    )

    for paths in [10_000, 25_000, 50_000, 100_000]:
        t0 = time.perf_counter()
        price, delta, p_se, d_se = price_and_delta_bermudan_put(
            S0=S0, K=K, r=r, q=q, sigma=sigma, T=T,
            eps_rel=1e-4,
            paths=paths,
            **cfg,
        )
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"paths={paths:>7d}  time={dt:>7.3f}s  price={price:.4f}  delta={delta:.4f}  p_se={p_se:.3e}  d_se={d_se:.3e}")

if __name__ == "__main__":
    main()
