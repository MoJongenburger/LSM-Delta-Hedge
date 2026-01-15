#include "src/cpp/lsm/lsm_model.h"
#include <cassert>
#include <iostream>

int main() {
    lsm::LSMConfig cfg;
    cfg.steps = 252;
    cfg.paths = 20000;         // smaller for quick smoke test
    cfg.seed = 123;
    cfg.basis_degree = 2;
    cfg.antithetic = true;

    const double S0 = 100.0;
    const double r  = 0.04;
    const double q  = 0.0;
    const double sig = 0.20;
    const double T = 1.0;

    // Put should have positive value
    auto p_atm = lsm::price_bermudan_put_lsm(S0, 100.0, r, q, sig, T, cfg);
    std::cout << "ATM Bermudan put price: " << p_atm.price
              << " (stderr " << p_atm.mc_stderr << ")\n";
    assert(p_atm.price >= 0.0);

    // Higher strike => higher put price
    auto p_hiK = lsm::price_bermudan_put_lsm(S0, 110.0, r, q, sig, T, cfg);
    std::cout << "K=110 Bermudan put price: " << p_hiK.price << "\n";
    assert(p_hiK.price >= p_atm.price);

    // Deep ITM put should be close-ish to intrinsic (not strict due to time value)
    auto p_itm = lsm::price_bermudan_put_lsm(S0, 150.0, r, q, sig, T, cfg);
    std::cout << "K=150 Bermudan put price: " << p_itm.price << "\n";
    assert(p_itm.price > 40.0); // intrinsic is 50; Bermudan put should be substantial

    std::cout << "Smoke test passed.\n";
    return 0;
}

