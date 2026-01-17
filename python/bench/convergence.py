from __future__ import annotations

import os, sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON_DIR = os.path.join(REPO_ROOT, "python")
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from lsmhedge import price_bermudan_put


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
    )

    for paths in [5_000, 10_000, 20_000, 40_000, 80_000]:
        p, se = price_bermudan_put(
            S0=S0, K=K, r=r, q=q, sigma=sigma, T=T,
            paths=paths,
            seed=123,
            **cfg,
        )
        print(f"paths={paths:>7d}  price={p:.5f}  stderr={se:.3e}")

if __name__ == "__main__":
    main()
