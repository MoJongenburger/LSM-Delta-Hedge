#pragma once

#include <cstdint>

namespace lsm {

enum class BasisType {
    Monomial,
    Laguerre
};

struct LSMConfig {
    int steps = 252;
    int paths = 100000;
    double train_fraction = 0.5;
    std::uint64_t seed = 42;

    int basis_degree = 2;
    BasisType basis = BasisType::Laguerre;

    bool antithetic = true;

    double ridge = 1e-8;

    bool use_control_variate = true;
};

struct LSMPriceResult {
    double price = 0.0;
    double mc_stderr = 0.0;
};

struct LSMPriceDeltaResult {
    double price = 0.0;        // time-0 price on test set (possibly CV-adjusted)
    double delta = 0.0;        // CRN finite-diff delta, using frozen regression betas
    double price_stderr = 0.0; // MC stderr of price estimator
    double delta_stderr = 0.0; // MC stderr of delta estimator
};

// Step 1 function (price only)
LSMPriceResult price_bermudan_put_lsm(
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    const LSMConfig& cfg
);

// Step 3 function (price + delta, no retraining for bumps)
LSMPriceDeltaResult price_and_delta_bermudan_put_lsm(
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    double eps_rel,      // e.g. 1e-4 => eps = eps_rel * S0
    const LSMConfig& cfg
);

} // namespace lsm
