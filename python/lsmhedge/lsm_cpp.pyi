from typing import Final

class BasisType:
    Monomial: Final[int]
    Laguerre: Final[int]

class LSMConfig:
    steps: int
    paths: int
    train_fraction: float
    seed: int
    basis_degree: int
    basis: int
    antithetic: bool
    ridge: float
    use_control_variate: bool

class LSMPriceResult:
    price: float
    mc_stderr: float

def price_bermudan_put_lsm(S0: float, K: float, r: float, q: float, sigma: float, T: float, cfg: LSMConfig) -> LSMPriceResult: ...
