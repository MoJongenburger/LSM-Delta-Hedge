#pragma once

#include "src/cpp/lsm/lsm_model.h"
#include <memory>

namespace lsm {

/**
 * Stateful LSM model:
 * - trains regression (betas) once at construction
 * - supports fast quotes at arbitrary start_step using stored test paths and betas
 *
 * Intended desk workflow:
 * - rebuild model weekly or when sigma/r changes materially
 * - on each day, call quote(S_today, start_step=days_since_rebuild)
 */
class LSMModel {
public:
    LSMModel(double K, double r, double q, double sigma, double T, const LSMConfig& cfg);
    ~LSMModel();

    LSMModel(const LSMModel&) = delete;
    LSMModel& operator=(const LSMModel&) = delete;

    LSMModel(LSMModel&&) noexcept;
    LSMModel& operator=(LSMModel&&) noexcept;

    // Quote value/delta at time index start_step (0 = model start)
    LSMPriceDeltaResult quote(double S0, int start_step = 0, double eps_rel = 1e-4) const;

    // Accessors
    int steps() const;
    double dt() const;
    double K() const;
    double r() const;
    double q() const;
    double sigma() const;
    double T() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace lsm
