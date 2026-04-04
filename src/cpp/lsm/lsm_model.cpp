#include "lsm_model.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace lsm {
namespace {

inline double put_payoff(double S, double K) { return std::max(K - S, 0.0); }

inline double put_delta_from_intrinsic(double S, double K) {
    return (S < K ? -1.0 : 0.0);
}

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
// L0 = 1, L1 = 1 - x, (n+1)L_{n+1} = (2n+1-x)L_n - nL_{n-1}
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

inline void init_terminal_scaled(const Eigen::MatrixXd& S_base, double scale, int steps, double K, WorkState& st) {
    const int paths = static_cast<int>(S_base.rows());
    st.cf.assign(paths, 0.0);
    st.tau.assign(paths, steps);
    for (int p = 0; p < paths; ++p) {
        st.cf[p] = put_payoff(scale * S_base(p, steps), K);
        st.tau[p] = steps;
    }
}

struct Policy {
    std::vector<Eigen::VectorXd> beta; // beta[n] is regression coeff at time n (size deg+1)
    std::vector<char> has_beta;        // has_beta[n] = 1 if beta[n] valid
};

Policy train_policy_lsm(
    const Eigen::MatrixXd& S_train,
    double K,
    int steps,
    const std::vector<double>& disc_step,
    int deg,
    BasisType basis,
    double ridge
) {
    const int n_train = static_cast<int>(S_train.rows());

    WorkState tr;
    init_terminal_scaled(S_train, 1.0, steps, K, tr);

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

            Eigen::RowVectorXd b(deg + 1);
            fill_basis(b, Sn / K, deg, basis);
            A.row(row) = b;

            const int tau_p = tr.tau[p];
            const int k = tau_p - n;
            y(row) = tr.cf[p] * disc_step[k];
        }

        Eigen::VectorXd beta = ridge_qr(A, y, ridge);
        pol.beta[n] = beta;
        pol.has_beta[n] = 1;

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

std::vector<double> apply_policy_to_test_scaled(
    const Eigen::MatrixXd& S_test_base,
    double scale,
    const Policy& pol,
    double K,
    int steps,
    const std::vector<double>& disc_step,
    int deg,
    BasisType basis,
    bool /*allow_start_exercise*/
) {
    const int n_test = static_cast<int>(S_test_base.rows());

    WorkState te;
    init_terminal_scaled(S_test_base, scale, steps, K, te);

    for (int n = steps - 1; n >= 1; --n) {
        if (!pol.has_beta[n]) continue;

        const Eigen::VectorXd& beta = pol.beta[n];
        for (int p = 0; p < n_test; ++p) {
            const double Sn = scale * S_test_base(p, n);
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

struct CVAdjustmentResult {
    std::vector<double> adjusted;
    double mean = 0.0;
    double stderr = 0.0;
};

CVAdjustmentResult maybe_apply_control_variate(
    const std::vector<double>& X,
    const Eigen::MatrixXd& S_test,
    double S0,
    double K,
    double r,
    double q,
    double sigma,
    double T,
    bool use_control_variate
) {
    CVAdjustmentResult out;
    out.adjusted = X;
    out.mean = mean_of(out.adjusted);

    if (!use_control_variate) {
        out.stderr = stderr_of(out.adjusted, out.mean);
        return out;
    }

    const int n_test = static_cast<int>(S_test.rows());
    const double bs = black_scholes_put(S0, K, r, q, sigma, T);
    const double discT = std::exp(-r * T);

    std::vector<double> Y(n_test, 0.0);
    for (int p = 0; p < n_test; ++p) {
        Y[p] = discT * put_payoff(S_test(p, S_test.cols() - 1), K);
    }

    const double mx = mean_of(X);
    const double my = mean_of(Y);

    double cov = 0.0;
    double vary = 0.0;
    for (int p = 0; p < n_test; ++p) {
        cov  += (X[p] - mx) * (Y[p] - my);
        vary += (Y[p] - my) * (Y[p] - my);
    }
    cov /= static_cast<double>(std::max(n_test - 1, 1));
    vary /= static_cast<double>(std::max(n_test - 1, 1));

    if (vary > 0.0) {
        const double beta_cv = cov / vary;
        for (int p = 0; p < n_test; ++p) {
            out.adjusted[p] = X[p] - beta_cv * (Y[p] - bs);
        }
        out.mean = mean_of(out.adjusted);
    }

    out.stderr = stderr_of(out.adjusted, out.mean);
    return out;
}

} // namespace

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

    Policy pol = train_policy_lsm(S_train, K, steps, disc_step, deg, cfg.basis, cfg.ridge);
    std::vector<double> X = apply_policy_to_test_scaled(
        S_test, 1.0, pol, K, steps, disc_step, deg, cfg.basis, true
    );

    CVAdjustmentResult cv = maybe_apply_control_variate(X, S_test, S0, K, r, q, sigma, T, cfg.use_control_variate);

    LSMPriceResult res;
    res.price = cv.mean;
    res.mc_stderr = cv.stderr;
    return res;
}

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
    if (eps_rel < 1e-6) throw std::invalid_argument("eps_rel too small (< 1e-6)");
    if (eps_rel > 1e-2) throw std::invalid_argument("eps_rel too large (> 1e-2)");

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

    Policy pol = train_policy_lsm(S_train, K, steps, disc_step, deg, cfg.basis, cfg.ridge);

    const double intrinsic = put_payoff(S0, K);

    // Continuation-only estimate at the current quote time.
    std::vector<double> X_cont = apply_policy_to_test_scaled(
        S_test, 1.0, pol, K, steps, disc_step, deg, cfg.basis, true
    );
    CVAdjustmentResult cv_cont = maybe_apply_control_variate(
        X_cont, S_test, S0, K, r, q, sigma, T, cfg.use_control_variate
    );

    const double continuation = cv_cont.mean;
    const bool exercise_now = intrinsic > continuation;

    LSMPriceDeltaResult out;
    out.intrinsic_value = intrinsic;
    out.continuation_value = continuation;
    out.exercise_now = exercise_now;

    if (exercise_now || T <= 0.0) {
        out.price = intrinsic;
        out.delta = put_delta_from_intrinsic(S0, K);
        out.price_stderr = 0.0;
        out.delta_stderr = 0.0;
        return out;
    }

    out.price = continuation;
    out.price_stderr = cv_cont.stderr;

    const double eps = eps_rel * S0;
    const double S_up = S0 + eps;
    const double S_dn = S0 - eps;
    if (S_dn <= 0.0) throw std::invalid_argument("epsilon too large: S0 - eps must be > 0");

    const double scale_up = S_up / S0;
    const double scale_dn = S_dn / S0;

    std::vector<double> Xup = apply_policy_to_test_scaled(
        S_test, scale_up, pol, K, steps, disc_step, deg, cfg.basis, true
    );
    std::vector<double> Xdn = apply_policy_to_test_scaled(
        S_test, scale_dn, pol, K, steps, disc_step, deg, cfg.basis, true
    );

    std::vector<double> di(n_test, 0.0);
    for (int p = 0; p < n_test; ++p) di[p] = (Xup[p] - Xdn[p]) / (2.0 * eps);

    out.delta = mean_of(di);
    out.delta_stderr = stderr_of(di, out.delta);
    return out;
}

} // namespace lsm
