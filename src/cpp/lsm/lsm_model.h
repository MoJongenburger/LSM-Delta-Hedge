#pragma once

#include <cstdint>

namespace lsm {

enum class BasisType {
    Monomial,   // [1, x, x^2, ...]
    Laguerre    // Laguerre polynomials (better conditioned)
};

struct LSMConfig {
    int steps = 252;                 // Daily grid for ~1y
    int paths = 100000;              // Total paths = train + test
    double train_fraction = 0.5;     // 0<frac<1; used for out-of-sample regression
    std::uint64_t seed = 42;

    int basis_degree = 2;
    BasisType basis = BasisType::Laguerre;

    bool antithetic = true;

    // Ridge regression strength (Tikhonov). Use ~1e-8 for stability.
    double ridge = 1e-8;

    // Optional European-put control variate for variance reduction.
    bool use_control_variate = true;
};

struct LSMPriceResult {
    double price = 0.0;
    double mc_stderr = 0.0;
};

LSMPriceResult price_bermudan_put_lsm(
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    const LSMConfig& cfg
);

} // namespace lsm

