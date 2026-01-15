#include "src/cpp/lsm/lsm_model.h"
#include <cassert>
#include <iostream>

int main() {
    lsm::LSMConfig cfg;
    cfg.steps = 252;
    cfg.paths = 20000;         // quick smoke
    cfg.seed = 123;
    cfg.basis_degree = 2;
    cfg.antithetic = true;

    const double S0 = 100.0;
    const double r  = 0.04;
    const double q  = 0.0;
    const double sig = 0.20;
    const double T = 1.0;

    auto p_atm = lsm::price_bermudan_put_lsm(S0, 100.0, r, q, sig, T, cfg);
    std::cout << "ATM Bermudan put price: " << p_atm.price
              << " (stderr " << p_atm.mc_stderr << ")\n";
    assert(p_atm.price >= 0.0);

    auto p_hiK = lsm::price_bermudan_put_lsm(S0, 110.0, r, q, sig, T, cfg);
    std::cout << "K=110 Bermudan put price: " << p_hiK.price << "\n";
    assert(p_hiK.price >= p_atm.price);

    auto p_itm = lsm::price_bermudan_put_lsm(S0, 150.0, r, q, sig, T, cfg);
    std::cout << "K=150 Bermudan put price: " << p_itm.price << "\n";
    assert(p_itm.price > 40.0);

    // Delta sanity: put delta should be negative; deeper ITM => more negative
    {
        lsm::LSMConfig cfg2 = cfg;
        cfg2.paths = 40000;

        auto atm = lsm::price_and_delta_bermudan_put_lsm(100.0, 100.0, r, q, sig, T, 1e-4, cfg2);
        auto itm2 = lsm::price_and_delta_bermudan_put_lsm(100.0, 150.0, r, q, sig, T, 1e-4, cfg2);

        std::cout << "ATM delta: " << atm.delta << " (se " << atm.delta_stderr << ")\n";
        std::cout << "ITM delta: " << itm2.delta << " (se " << itm2.delta_stderr << ")\n";

        assert(atm.delta <= 0.0);
        assert(itm2.delta <= 0.0);
        assert(itm2.delta < atm.delta);
        assert(itm2.delta > -1.0);
    }

    // Delta stderr should improve with more paths (rough check)
    {
        lsm::LSMConfig c1 = cfg;
        c1.paths = 20000;
        auto r1 = lsm::price_and_delta_bermudan_put_lsm(100.0, 100.0, r, q, sig, T, 1e-4, c1);

        lsm::LSMConfig c2 = cfg;
        c2.paths = 80000;
        auto r2 = lsm::price_and_delta_bermudan_put_lsm(100.0, 100.0, r, q, sig, T, 1e-4, c2);

        std::cout << "delta stderr N=20k: " << r1.delta_stderr << "\n";
        std::cout << "delta stderr N=80k: " << r2.delta_stderr << "\n";
        assert(r2.delta_stderr < 0.8 * r1.delta_stderr);
    }

    std::cout << "Smoke test passed.\n";
    return 0;
}
