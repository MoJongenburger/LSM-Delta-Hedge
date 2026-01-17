from __future__ import annotations
import math


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_put(S: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    if T <= 0.0:
        return max(K - S, 0.0)
    if sigma <= 0.0:
        ST = S * math.exp((r - q) * T)
        return math.exp(-r * T) * max(K - ST, 0.0)

    vol = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
