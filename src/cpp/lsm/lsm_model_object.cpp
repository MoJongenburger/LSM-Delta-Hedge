#include "src/cpp/lsm/lsm_model_object.h"

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

// Ridge via augmented QR
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
            S(p, n) = S(p, n - 1) * std::exp(drift + vol * z);

            if (use_anti) {
                const int pa = p + half;
                if (pa < paths) {
                    S(pa, n) = S(pa, n - 1) * std::exp(drift + vol * (-z));
                }
            }
        }
    }
    return S;
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

struct LSMModel::Impl {
    struct WorkState {
        std::vector<double> cf;
        std::vector<int> tau;
    };

    struct Policy {
        std::vector<Eigen::VectorXd> beta;
        std::vector<char> has_beta;
    };

    double K = 0.0, r = 0.0, q = 0.0, sigma = 0.0, T = 0.0;
    LSMConfig cfg{};
    int steps = 0;
    double dt = 0.0;

    int n_train = 0;
    int n_test = 0;

    std::vector<double> disc_step; // exp(-r*dt*k)

    Eigen::MatrixXd S_train; // (n_train x steps+1), simulated with S0=K
    Eigen::MatrixXd S_test;  // (n_test x steps+1), simulated with S0=K

    Policy pol;

    Policy train_policy_lsm(const Eigen::MatrixXd& S_tr) {
        const int deg = cfg.basis_degree;
        const int paths = static_cast<int>(S_tr.rows());

        WorkState st;
        st.cf.assign(paths, 0.0);
        st.tau.assign(paths, steps);

        // terminal payoff
        for (int p = 0; p < paths; ++p) {
            st.cf[p] = put_payoff(S_tr(p, steps), K);
            st.tau[p] = steps;
        }

        Policy out;
        out.beta.resize(steps + 1);
        out.has_beta.assign(steps + 1, 0);

        for (int n = steps - 1; n >= 1; --n) {
            std::vector<int> itm;
            itm.reserve(paths);
            for (int p = 0; p < paths; ++p) {
                if (put_payoff(S_tr(p, n), K) > 0.0) itm.push_back(p);
            }

            const int m = static_cast<int>(itm.size());
            if (m <= (deg + 1)) continue;

            Eigen::MatrixXd A(m, deg + 1);
            Eigen::VectorXd y(m);

            for (int row = 0; row < m; ++row) {
                const int p = itm[row];
                const double Sn = S_tr(p, n);

                Eigen::RowVectorXd b(deg + 1);
                fill_basis(b, Sn / K, deg, cfg.basis);
                A.row(row) = b;

                const int k = st.tau[p] - n;
                y(row) = st.cf[p] * disc_step[k];
            }

            Eigen::VectorXd beta = ridge_qr(A, y, cfg.ridge);
            out.beta[n] = beta;
            out.has_beta[n] = 1;

            // update exercise decisions on training set
            for (int row = 0; row < m; ++row) {
                const int p = itm[row];
                const double Sn = S_tr(p, n);
                const double exercise = put_payoff(Sn, K);

                Eigen::RowVectorXd b(deg + 1);
                fill_basis(b, Sn / K, deg, cfg.basis);
                const double continuation = (b * beta)(0);

                if (exercise > continuation) {
                    st.cf[p] = exercise;
                    st.tau[p] = n;
                }
            }
        }

        return out;
    }

    // Apply stored policy to test paths, quoting at start_step for observed spot S0:
    // S_m = S0 * (S_test(p,m) / S_test(p,start_step))
    std::vector<double> apply_policy_test(double S0, int start_step) const {
        const int deg = cfg.basis_degree;
        const int paths = n_test;

        WorkState st;
        st.cf.assign(paths, 0.0);
        st.tau.assign(paths, steps);

        std::vector<double> inv_base(paths, 0.0);
        for (int p = 0; p < paths; ++p) {
            const double base = S_test(p, start_step);
            inv_base[p] = 1.0 / base;
        }

        // terminal payoff at maturity, discounted back to start_step later
        for (int p = 0; p < paths; ++p) {
            const double ST = S0 * S_test(p, steps) * inv_base[p];
            st.cf[p] = put_payoff(ST, K);
            st.tau[p] = steps;
        }

        const int n0 = std::max(1, start_step);
        for (int n = steps - 1; n >= n0; --n) {
            if (!pol.has_beta[n]) continue;
            const Eigen::VectorXd& beta = pol.beta[n];

            for (int p = 0; p < paths; ++p) {
                const double Sn = S0 * S_test(p, n) * inv_base[p];
                const double exercise = put_payoff(Sn, K);
                if (exercise <= 0.0) continue;

                Eigen::RowVectorXd b(deg + 1);
                fill_basis(b, Sn / K, deg, cfg.basis);
                const double continuation = (b * beta)(0);

                if (exercise > continuation) {
                    st.cf[p] = exercise;
                    st.tau[p] = n;
                }
            }
        }

        std::vector<double> X(paths, 0.0);
        for (int p = 0; p < paths; ++p) {
            const int k = st.tau[p] - start_step;
            X[p] = st.cf[p] * disc_step[k];
        }
        return X;
    }

    LSMPriceDeltaResult quote(double S0, int start_step, double eps_rel) const {
        if (!(S0 > 0.0)) throw std::invalid_argument("S0 must be > 0");
        if (start_step < 0 || start_step > steps) throw std::invalid_argument("start_step out of range");
        if (!(eps_rel > 0.0)) throw std::invalid_argument("eps_rel must be > 0");
        if (eps_rel < 1e-6) throw std::invalid_argument("eps_rel too small (< 1e-6)");
        if (eps_rel > 1e-2) throw std::invalid_argument("eps_rel too large (> 1e-2)");

        // expiry
        if (start_step == steps) {
            const double payoff = put_payoff(S0, K);
            LSMPriceDeltaResult out;
            out.price = payoff;
            out.delta = (S0 < K ? -1.0 : 0.0);
            out.price_stderr = 0.0;
            out.delta_stderr = 0.0;
            return out;
        }

        const double eps = eps_rel * S0;
        const double Sup = S0 + eps;
        const double Sdn = S0 - eps;
        if (Sdn <= 0.0) throw std::invalid_argument("epsilon too large: S0 - eps must be > 0");

        std::vector<double> X0 = apply_policy_test(S0, start_step);

        // Optional control variate at this start_step
        std::vector<double> X0_used = X0;
        double price0 = mean_of(X0);

        if (cfg.use_control_variate) {
            const double T_rem = (steps - start_step) * dt;
            const double bs = black_scholes_put(S0, K, r, q, sigma, T_rem);
            const double discT = std::exp(-r * T_rem);

            // maturity payoff discounted to start_step
            std::vector<double> Y(n_test, 0.0);
            for (int p = 0; p < n_test; ++p) {
                const double base = S_test(p, start_step);
                const double inv = 1.0 / base;
                const double ST = S0 * S_test(p, steps) * inv;
                Y[p] = discT * put_payoff(ST, K);
            }

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
                for (int p = 0; p < n_test; ++p) {
                    X0_used[p] = X0[p] - beta_cv * (Y[p] - bs);
                }
                price0 = mean_of(X0_used);
            }
        }

        const double price_se = stderr_of(X0_used, price0);

        // Delta via CRN bumps (policy re-applied on same test paths)
        std::vector<double> Xup = apply_policy_test(Sup, start_step);
        std::vector<double> Xdn = apply_policy_test(Sdn, start_step);

        std::vector<double> di(n_test, 0.0);
        for (int p = 0; p < n_test; ++p) {
            di[p] = (Xup[p] - Xdn[p]) / (2.0 * eps);
        }

        const double delta = mean_of(di);
        const double delta_se = stderr_of(di, delta);

        LSMPriceDeltaResult out;
        out.price = price0;
        out.delta = delta;
        out.price_stderr = price_se;
        out.delta_stderr = delta_se;
        return out;
    }
};

LSMModel::LSMModel(double K, double r, double q, double sigma, double T, const LSMConfig& cfg)
: impl_(std::make_unique<Impl>()) {
    if (!(K > 0.0)) throw std::invalid_argument("K must be > 0");
    if (!(T > 0.0)) throw std::invalid_argument("T must be > 0");
    if (!(sigma >= 0.0)) throw std::invalid_argument("sigma must be >= 0");
    if (cfg.steps <= 0) throw std::invalid_argument("cfg.steps must be > 0");
    if (cfg.paths <= 100) throw std::invalid_argument("cfg.paths too small for model object");
    if (!(cfg.train_fraction > 0.0 && cfg.train_fraction < 1.0))
        throw std::invalid_argument("cfg.train_fraction must be in (0,1)");

    impl_->K = K;
    impl_->r = r;
    impl_->q = q;
    impl_->sigma = sigma;
    impl_->T = T;
    impl_->cfg = cfg;
    impl_->steps = cfg.steps;
    impl_->dt = T / static_cast<double>(cfg.steps);

    impl_->n_train = std::max(100, static_cast<int>(std::floor(cfg.paths * cfg.train_fraction)));
    impl_->n_test  = std::max(100, cfg.paths - impl_->n_train);

    impl_->disc_step.assign(impl_->steps + 1, 1.0);
    for (int k = 1; k <= impl_->steps; ++k) {
        impl_->disc_step[k] = std::exp(-r * impl_->dt * static_cast<double>(k));
    }

    // simulate with S0 = K to keep basis x = S/K around 1 at start
    impl_->S_train = simulate_gbm(impl_->n_train, impl_->steps, K, r, q, sigma, T, cfg.seed, cfg.antithetic);
    impl_->S_test  = simulate_gbm(impl_->n_test,  impl_->steps, K, r, q, sigma, T, cfg.seed + 1, cfg.antithetic);

    impl_->pol = impl_->train_policy_lsm(impl_->S_train);
}

LSMModel::~LSMModel() = default;
LSMModel::LSMModel(LSMModel&&) noexcept = default;
LSMModel& LSMModel::operator=(LSMModel&&) noexcept = default;

LSMPriceDeltaResult LSMModel::quote(double S0, int start_step, double eps_rel) const {
    return impl_->quote(S0, start_step, eps_rel);
}

int LSMModel::steps() const { return impl_->steps; }
double LSMModel::dt() const { return impl_->dt; }
double LSMModel::K() const { return impl_->K; }
double LSMModel::r() const { return impl_->r; }
double LSMModel::q() const { return impl_->q; }
double LSMModel::sigma() const { return impl_->sigma; }
double LSMModel::T() const { return impl_->T; }

} // namespace lsm
