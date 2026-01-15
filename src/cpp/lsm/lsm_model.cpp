#include "lsm_model.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace lsm {
namespace {

inline double put_payoff(double S, double K) { return std::max(K - S, 0.0); }

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

double black_scholes_put(double S, double K, double r, double q, double sigma, double T) {
    if (T <= 0.0) return put_payoff(S, K);
    if (sigma <= 0.0) {
        const double ST = S * std::exp((r - q) * T);
        return std::exp(-r * T) * put_payoff(ST, K);
    }
    const double vol_sqrtT = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT;
    const double d2 = d1 - vol_sqrtT;
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
}

inline void fill_monomial(Eigen::RowVectorXd& row, double x, int deg) {
    row(0) = 1.0;
    double v = 1.0;
    for (int j = 1; j <= deg; ++j) {
        v *= x;
        row(j) = v;
    }
}

// Laguerre L0..Ldeg with recurrence:
// L0=1, L1=1-x, (n+1)L_{n+1} = (2n+1-x)L_n - nL_{n-1}
inline void fill_laguerre(Eigen::RowVectorXd& row, double x, int deg) {
    row(0) = 1.0;
    if (deg == 0) return;
    row(1) = 1.0 - x;
    if (deg == 1) return;

    double Lnm1 = row(0);
    double Ln = row(1);
    for (int n = 1; n < deg; ++n) {
        const double Lnp1 = ((2.0 * n + 1.0 - x) * Ln - n * Lnm1) / (n + 1.0);
        row(n + 1) = Lnp1;
        Lnm1 = Ln;
        Ln = Lnp1;
    }
}

inline void fill_basis(Eigen::RowVectorXd& row, double x, int deg, BasisType bt) {
    if (bt == BasisType::Monomial) fill_monomial(row, x, deg);
    else fill_laguerre(row, x, deg);
}

// Ridge via augmented QR: min ||A b - y||^2 + lambda ||b||^2
Eigen::VectorXd ridge_qr(const Eigen::MatrixXd& A, const Eigen::VectorXd& y, double lambda) {
    if (lambda <= 0.0) return A.colPivHouseholderQr().solve(y);

    const int m = static_cast<int>(A.rows());
    const int k = static_cast<int>(A.cols());

    Eigen::MatrixXd Aaug(m + k, k);
    Eigen::VectorXd yaug(m + k);
    Aaug.topRows(m) = A;
    yaug.head(m) = y;

    Aaug.bottomRows(k).setZero();
    Aaug.bottomRows(k).diagonal().array() = std::sqrt(lambda);
    yaug.tail(k).setZero();

    return Aaug.colPivHouseholderQr().solve(yaug);
}

// Simulate GBM paths into Eigen matrix (paths x (steps+1))
Eigen::MatrixXd simulate_gbm(
    int paths,
    int steps,
    double S0,
    double r,
    double q,
    double sigma,
    double T,
    std::uint64_t seed,
    bool antithetic
) {
    if (paths <= 0) throw std::invalid_argument("paths must be > 0");
    if (steps <= 0) throw std::invalid_argument("steps must be > 0");
    if (!(S0 > 0.0)) throw std::invalid_argument("S0 must be > 0");
    if (!(T > 0.0)) throw std::invalid_argument("T must be > 0");
    if (!(sigma >= 0.0)) throw std::invalid_argument("sigma must be >= 0");

    const double dt = T / static_cast<double>(steps);
    const double drift = (r - q - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    Eigen::MatrixXd S(paths, steps + 1);
    S.col(0).setConstant(S0);

    const bool use_anti = antithetic && (paths >= 2);
    const int half = use_anti ? (paths / 2) : paths;

    for (int n = 1; n <= steps; ++n) {
        for (int p = 0; p < half; ++p) {
            const double z = nd(rng);
            const double mult = std::exp(drift + vol * z);
            S(p, n) = S(p, n - 1) * mult;

            if (use_anti) {
                const int pa = p + half;
                if (pa < paths) {
                    const double mult_a = std::exp(drift + vol * (-z));
                    S(pa, n) = S(pa, n - 1) * mult_a;
                }
            }
        }
    }
    return S;
}

struct WorkState {
    std::vector<double> cf;  // cashflow at exercise time
    std::vector<int> tau;    // exercise index in [0..steps]
};

inline void init_terminal(const Eigen::MatrixXd& S, int steps, double K, WorkState& st) {
    const int paths = static_cast<int>(S.rows());
    st.cf.assign(paths, 0.0);
    st.tau.assign(paths, steps);
    for (int p = 0; p < paths; ++p) {
        st.cf[p] = put_payoff(S(p, steps), K);
        st.tau[p] = steps;
    }
}

struct Policy {
    std::vector<Eigen::VectorXd> beta; // beta[n] is regression coeff at time n (size deg+1)
    std::vector<char> has_beta;        // has_beta[n]=1 if beta[n] valid
};

// Train LSM continuation regressions on training paths (and update train exercise decisions).
Policy train_policy_lsm(
    const Eigen::MatrixXd& S_train,
    double K,
    int steps,
    double r,
    double dt,
    const std::vector<double>& disc_step,
    int deg,
    BasisType basis,
    double ridge
) {
    const int n_train = static_cast<int>(S_train.rows());

    WorkState tr;
    init_terminal(S_train, steps, K, tr);

    Policy pol;
    pol.beta.resize(steps + 1);
    pol.has_beta.assign(steps + 1, 0);

    for (int n = steps - 1; n >= 1; --n) {
        std::vector<int> itm;
        itm.reserve(n_train);
        for (int p = 0; p < n_train; ++p) {
            if (put_payoff(S_train(p, n), K) > 0.0) itm.push_back(p);
        }

        const int m = static_cast<int>(itm.size());
        if (m <= (deg + 1)) continue;

        Eigen::MatrixXd A(m, deg + 1);
        Eigen::VectorXd y(m);

        for (int row = 0; row < m; ++row) {
            const int p = itm[row];
            const double Sn = S_train(p, n);
            const double x = Sn / K;

            Eigen::RowVectorXd b(deg + 1);
            fill_basis(b, x, deg, basis);
            A.row(row) = b;

            const int tau_p = tr.tau[p];
            const int k = tau_p - n;
            y(row) = tr.cf[p] * disc_step[k];
        }

        Eigen::VectorXd beta = ridge_qr(A, y, ridge);
        pol.beta[n] = beta;
        pol.has_beta[n] = 1;

        // Update train exercise decisions using the trained beta
        for (int row = 0; row < m; ++row) {
            const int p = itm[row];
            const double Sn = S_train(p, n);
            const double exercise = put_payoff(Sn, K);

            Eigen::RowVectorXd b(deg + 1);
            fill_basis(b, Sn / K, deg, basis);
            const double continuation = (b * beta)(0);

            if (exercise > continuation) {
                tr.cf[p] = exercise;
                tr.tau[p] = n;
            }
        }
    }

    return pol;
}

// Apply a trained policy to a set of test paths to produce discounted time-0 cashflows X[p]
std::vector<double> apply_policy_to_test(
    const Eigen::MatrixXd& S_test,
    const Policy& pol,
    double K,
    int steps,
    double dt,
    const std::vector<double>& disc_step,
    int deg,
    BasisType basis
) {
    const int n_test = static_cast<int>(S_test.rows());
    WorkState te;
    init_terminal(S_test, steps, K, te);

    for (int n = steps - 1; n >= 1; --n) {
        if (!pol.has_beta[n]) continue;
        const Eigen::VectorXd& beta = pol.beta[n];

        for (int p = 0; p < n_test; ++p) {
            const double Sn = S_test(p, n);
            const double exercise = put_payoff(Sn, K);
            if (exercise <= 0.0) continue;

            Eigen::RowVectorXd b(deg + 1);
            fill_basis(b, Sn / K, deg, basis);
            const double continuation = (b * beta)(0);

            if (exercise > continuation) {
                te.cf[p] = exercise;
                te.tau[p] = n;
            }
        }
    }

    std::vector<double> X(n_test, 0.0);
    for (int p = 0; p < n_test; ++p) {
        X[p] = te.cf[p] * disc_step[te.tau[p]];
    }
    return X;
}

inline double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

inline double stderr_of(const std::vector<double>& v, double mean) {
    const int n = static_cast<int>(v.size());
    double var = 0.0;
    for (double x : v) {
        const double d = x - mean;
        var += d * d;
    }
    var /= static_cast<double>(std::max(n - 1, 1));
    return std::sqrt(var / static_cast<double>(n));
}

} // namespace

// --------------------
// Step 1: price-only
// --------------------
LSMPriceResult price_bermudan_put_lsm(
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    const LSMConfig& cfg
) {
    if (!(S0 > 0.0)) throw std::invalid_argument("S0 must be > 0");
    if (!(K > 0.0)) throw std::invalid_argument("K must be > 0");
    if (!(T > 0.0)) throw std::invalid_argument("T must be > 0");
    if (cfg.steps <= 0) throw std::invalid_argument("cfg.steps must be > 0");
    if (cfg.paths <= 10) throw std::invalid_argument("cfg.paths must be reasonably large");
    if (!(cfg.train_fraction > 0.0 && cfg.train_fraction < 1.0))
        throw std::invalid_argument("cfg.train_fraction must be in (0,1)");
    if (cfg.basis_degree < 0) throw std::invalid_argument("cfg.basis_degree must be >= 0");

    const int steps = cfg.steps;
    const int deg = cfg.basis_degree;
    const double dt = T / static_cast<double>(steps);

    const int n_train = std::max(10, static_cast<int>(std::floor(cfg.paths * cfg.train_fraction)));
    const int n_test  = std::max(10, cfg.paths - n_train);

    std::vector<double> disc_step(steps + 1, 1.0);
    for (int k = 1; k <= steps; ++k) disc_step[k] = std::exp(-r * dt * static_cast<double>(k));

    Eigen::MatrixXd S_train = simulate_gbm(n_train, steps, S0, r, q, sigma, T, cfg.seed, cfg.antithetic);
    Eigen::MatrixXd S_test  = simulate_gbm(n_test,  steps, S0, r, q, sigma, T, cfg.seed + 1, cfg.antithetic);

    Policy pol = train_policy_lsm(S_train, K, steps, r, dt, disc_step, deg, cfg.basis, cfg.ridge);

    std::vector<double> X = apply_policy_to_test(S_test, pol, K, steps, dt, disc_step, deg, cfg.basis);

    double price = mean_of(X);

    // Optional control variate: European put payoff at maturity
    if (cfg.use_control_variate) {
        const double bs = black_scholes_put(S0, K, r, q, sigma, T);
        const double discT = std::exp(-r * T);

        std::vector<double> Y(n_test, 0.0);
        for (int p = 0; p < n_test; ++p) {
            Y[p] = discT * put_payoff(S_test(p, steps), K);
        }

        const double mx = mean_of(X);
        const double my = mean_of(Y);

        double cov = 0.0, vary = 0.0;
        for (int p = 0; p < n_test; ++p) {
            cov  += (X[p] - mx) * (Y[p] - my);
            vary += (Y[p] - my) * (Y[p] - my);
        }
        cov /= static_cast<double>(std::max(n_test - 1, 1));
        vary /= static_cast<double>(std::max(n_test - 1, 1));

        if (vary > 0.0) {
            const double beta_cv = cov / vary;
            std::vector<double> Xadj(n_test, 0.0);
            for (int p = 0; p < n_test; ++p) Xadj[p] = X[p] - beta_cv * (Y[p] - bs);

            price = mean_of(Xadj);

            LSMPriceResult res;
            res.price = price;
            res.mc_stderr = stderr_of(Xadj, price);
            return res;
        }
    }

    LSMPriceResult res;
    res.price = price;
    res.mc_stderr = stderr_of(X, price);
    return res;
}

// --------------------
// Step 3: price + delta (CRN, frozen regression)
// --------------------
LSMPriceDeltaResult price_and_delta_bermudan_put_lsm(
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    double eps_rel,
    const LSMConfig& cfg
) {
    if (!(S0 > 0.0)) throw std::invalid_argument("S0 must be > 0");
    if (!(K > 0.0)) throw std::invalid_argument("K must be > 0");
    if (!(T > 0.0)) throw std::invalid_argument("T must be > 0");
    if (!(eps_rel > 0.0)) throw std::invalid_argument("eps_rel must be > 0");
    if (cfg.steps <= 0) throw std::invalid_argument("cfg.steps must be > 0");
    if (cfg.paths <= 10) throw std::invalid_argument("cfg.paths must be reasonably large");
    if (!(cfg.train_fraction > 0.0 && cfg.train_fraction < 1.0))
        throw std::invalid_argument("cfg.train_fraction must be in (0,1)");
    if (cfg.basis_degree < 0) throw std::invalid_argument("cfg.basis_degree must be >= 0");

    const int steps = cfg.steps;
    const int deg = cfg.basis_degree;
    const double dt = T / static_cast<double>(steps);

    const int n_train = std::max(10, static_cast<int>(std::floor(cfg.paths * cfg.train_fraction)));
    const int n_test  = std::max(10, cfg.paths - n_train);

    std::vector<double> disc_step(steps + 1, 1.0);
    for (int k = 1; k <= steps; ++k) disc_step[k] = std::exp(-r * dt * static_cast<double>(k));

    // Base simulation (train/test) — CRN for bumps will reuse test shocks implicitly via scaling.
    Eigen::MatrixXd S_train = simulate_gbm(n_train, steps, S0, r, q, sigma, T, cfg.seed, cfg.antithetic);
    Eigen::MatrixXd S_test  = simulate_gbm(n_test,  steps, S0, r, q, sigma, T, cfg.seed + 1, cfg.antithetic);

    // Train once at base S0
    Policy pol = train_policy_lsm(S_train, K, steps, r, dt, disc_step, deg, cfg.basis, cfg.ridge);

    // Price base on test set
    std::vector<double> X0 = apply_policy_to_test(S_test, pol, K, steps, dt, disc_step, deg, cfg.basis);
    double price0 = mean_of(X0);

    // Optional control variate (base only)
    std::vector<double> X0_used = X0;
    if (cfg.use_control_variate) {
        const double bs = black_scholes_put(S0, K, r, q, sigma, T);
        const double discT = std::exp(-r * T);
        std::vector<double> Y(n_test, 0.0);
        for (int p = 0; p < n_test; ++p) Y[p] = discT * put_payoff(S_test(p, steps), K);

        const double mx = mean_of(X0);
        const double my = mean_of(Y);

        double cov = 0.0, vary = 0.0;
        for (int p = 0; p < n_test; ++p) {
            cov  += (X0[p] - mx) * (Y[p] - my);
            vary += (Y[p] - my) * (Y[p] - my);
        }
        cov /= static_cast<double>(std::max(n_test - 1, 1));
        vary /= static_cast<double>(std::max(n_test - 1, 1));

        if (vary > 0.0) {
            const double beta_cv = cov / vary;
            for (int p = 0; p < n_test; ++p) X0_used[p] = X0[p] - beta_cv * (Y[p] - bs);
            price0 = mean_of(X0_used);
        }
    }
    const double price_stderr = stderr_of(X0_used, price0);

    // CRN finite-diff delta with frozen regression:
    // Under GBM, paths scale linearly with S0 for fixed shocks => S_test_bump = S_test * (S0_bump/S0)
    const double eps = eps_rel * S0;
    const double S_up = S0 + eps;
    const double S_dn = S0 - eps;
    if (S_dn <= 0.0) throw std::invalid_argument("epsilon too large: S0 - eps must be > 0");

    const double scale_up = S_up / S0;
    const double scale_dn = S_dn / S0;

    Eigen::MatrixXd S_test_up = S_test * scale_up;
    Eigen::MatrixXd S_test_dn = S_test * scale_dn;

    // Apply same trained policy (betas) to bumped test paths
    std::vector<double> Xup = apply_policy_to_test(S_test_up, pol, K, steps, dt, disc_step, deg, cfg.basis);
    std::vector<double> Xdn = apply_policy_to_test(S_test_dn, pol, K, steps, dt, disc_step, deg, cfg.basis);

    // Delta estimator using per-path CRN differences (good variance reduction)
    std::vector<double> di(n_test, 0.0);
    for (int p = 0; p < n_test; ++p) di[p] = (Xup[p] - Xdn[p]) / (2.0 * eps);

    const double delta = mean_of(di);
    const double delta_stderr = stderr_of(di, delta);

    LSMPriceDeltaResult out;
    out.price = price0;
    out.delta = delta;
    out.price_stderr = price_stderr;
    out.delta_stderr = delta_stderr;
    return out;
}

} // namespace lsm
