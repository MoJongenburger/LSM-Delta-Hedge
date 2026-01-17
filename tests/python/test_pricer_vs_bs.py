import math
import numpy as np

from lsmhedge import price_bermudan_put
from lsmhedge.bs import bs_put


def test_european_put_matches_black_scholes_with_steps_1():
    S0 = 100.0
    K = 100.0
    r = 0.04
    q = 0.0
    sigma = 0.2
    T = 1.0

    # With steps=1, exercise only at maturity -> European put
    price_mc, se = price_bermudan_put(
        S0=S0, K=K, r=r, q=q, sigma=sigma, T=T,
        steps=1,
        paths=80_000,
        train_fraction=0.5,
        basis="Laguerre",
        basis_degree=2,
        seed=123,
        antithetic=True,
        use_control_variate=True,
    )

    price_bs = bs_put(S0, K, r, q, sigma, T)

    # Loose but meaningful tolerance: within ~4 standard errors
    assert abs(price_mc - price_bs) <= 4.0 * se + 0.05
