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

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double black_scholes_put(double S, double K, double r, double q, double sigma, double T) {
    if (T <= 0.0) return put_payoff(S, K);
    if (sigma <= 0.0) {
        // Deterministic forward under Q: S_T = S*exp((r-q)T)
        const double ST = S * std::exp((r - q) * T);
        return std::exp(-r * T) * put_payoff(ST, K);
    }
    const double vol_sqrtT = sigma * std::sqrt(T);
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT;
    const double d2 = d1 - vol_sqrtT;
    return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
}

// Basis builders on x = S/K (dimensionless)
inline void fill_monomial(Eigen::RowVectorXd& row, double x, int deg) {
    row(0) = 1.0;
    double v = 1.0;
    for (int j = 1; j <= deg; ++j) {
        v *= x;
        row(j) = v;
    }
}

// Laguerre polynomials L0..Ldeg (physicists' convention / common in LSM):
// L0(x)=1
// L1(x)=1-x
// recurrence: (n+1)L_{n+1} = (2n+1-x)L_n - n L_{n-1}
inline void fill_laguerre(Eigen::RowVectorXd& row, double x, int deg) {
    row(0) = 1.0;
    if (deg == 0) return;
    row(1) = 1.0 - x;
    if (deg == 1) return;

    double Lnm1 = row(0);
    double Ln   = row(1);
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

// Ridge regression via augmented QR (stable):
// minimize ||A b - y||^2 + lambda ||b||^2
Eigen::VectorXd ridge_qr(const Eigen::MatrixXd& A, const Eigen::VectorXd& y, double lambda) {
    if (lambda <= 0.0) {
        return A.colPivHouseholderQr().solve(y);
    }
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

// Simulate GBM paths into Eigen matrix (paths x (steps+1)), ColumnMajor by default.
// We access columns frequently for regression => ColumnMajor is a good default.
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
        // generate shocks
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
        if (use_anti && (paths % 2 == 1)) {
            // If odd, last path may not have been filled if pa==paths; ensure it's filled
            const int last = paths - 1;
            if (last >= half) {
                // already filled via pa<paths check above; nothing to do
            }
        }
    }

    return S;
}

struct WorkState {
    std::vector<double> cf;   // cashflow at exercise time
    std::vector<int> tau;     // exercise index (0..steps), default steps
};

void init_terminal(const Eigen::MatrixXd& S, int steps, double K, WorkState& st) {
    const int paths = static_cast<int>(S.rows());
    st.cf.assign(paths, 0.0);
    st.tau.assign(paths, steps);
    for (int p = 0; p < paths; ++p) {
        st.cf[p] = put_payoff(S(p, steps), K);
        st.tau[p] = steps;
    }
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

    // Split paths: train for regression, test for pricing/exercise decisions (out-of-sample)
    const int n_train = std::max(10, static_cast<int>(std::floor(cfg.paths * cfg.train_fraction)));
    const int n_test  = std::max(10, cfg.paths - n_train);

    const double dt = T / static_cast<double>(steps);

    // Precompute exp(-r * dt * k) for k=0..steps
    std::vector<double> disc_step(steps + 1, 1.0);
    for (int k = 1; k <= steps; ++k) {
        disc_step[k] = std::exp(-r * dt * static_cast<double>(k));
    }

    Eigen::MatrixXd S_train = simulate_gbm(n_train, steps, S0, r, q, sigma, T, cfg.seed, cfg.antithetic);
    Eigen::MatrixXd S_test  = simulate_gbm(n_test,  steps, S0, r, q, sigma, T, cfg.seed + 1, cfg.antithetic);

    WorkState tr, te;
    init_terminal(S_train, steps, K, tr);
    init_terminal(S_test,  steps, K, te);

    // Backward induction: n = steps-1 ... 1 (no exercise at n=0)
    for (int n = steps - 1; n >= 1; --n) {
        // Collect ITM indices in training set
        std::vector<int> itm_tr;
        itm_tr.reserve(n_train);
        for (int p = 0; p < n_train; ++p) {
            if (put_payoff(S_train(p, n), K) > 0.0) itm_tr.push_back(p);
        }

        const int m = static_cast<int>(itm_tr.size());
        if (m <= (deg + 1)) continue;

        Eigen::MatrixXd A(m, deg + 1);
        Eigen::VectorXd y(m);

        // Build regression (train): y = discounted CF back to time n
        for (int row = 0; row < m; ++row) {
            const int p = itm_tr[row];
            const double Sn = S_train(p, n);
            const double x = Sn / K;

            Eigen::RowVectorXd basis(deg + 1);
            fill_basis(basis, x, deg, cfg.basis);
            A.row(row) = basis;

            const int tau_p = tr.tau[p];
            const int k = tau_p - n;
            const double disc = disc_step[k];
            y(row) = tr.cf[p] * disc;
        }

        const Eigen::VectorXd beta = ridge_qr(A, y, cfg.ridge);

        // Apply exercise rule on TRAIN paths (to update their stopping decisions)
        for (int row = 0; row < m; ++row) {
            const int p = itm_tr[row];
            const double Sn = S_train(p, n);
            const double exercise = put_payoff(Sn, K);

            const double x = Sn / K;
            Eigen::RowVectorXd basis(deg + 1);
            fill_basis(basis, x, deg, cfg.basis);
            const double continuation = (basis * beta)(0);

            if (exercise > continuation) {
                tr.cf[p] = exercise;
                tr.tau[p] = n;
            }
        }

        // Apply SAME beta to TEST paths (out-of-sample exercise decisions)
        // Only evaluate for ITM test paths (efficiency)
        for (int p = 0; p < n_test; ++p) {
            const double Sn = S_test(p, n);
            const double exercise = put_payoff(Sn, K);
            if (exercise <= 0.0) continue;

            const double x = Sn / K;
            Eigen::RowVectorXd basis(deg + 1);
            fill_basis(basis, x, deg, cfg.basis);
            const double continuation = (basis * beta)(0);

            if (exercise > continuation) {
                te.cf[p] = exercise;
                te.tau[p] = n;
            }
        }
    }

    // Discount test CFs to time 0
    std::vector<double> X(n_test, 0.0);
    for (int p = 0; p < n_test; ++p) {
        X[p] = te.cf[p] * disc_step[te.tau[p]];
    }

    auto mean_of = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    };

    double price = mean_of(X);

    // Optional control variate using European put payoff at maturity
    // CV variable: Y = exp(-rT) * (K - S_T)+
    if (cfg.use_control_variate) {
        const double bs = black_scholes_put(S0, K, r, q, sigma, T);

        std::vector<double> Y(n_test, 0.0);
        const double discT = std::exp(-r * T);
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

            // Adjusted estimator:
            // X_adj = X - beta*(Y - E[Y]) where E[Y]=BS
            std::vector<double> Xadj(n_test, 0.0);
            for (int p = 0; p < n_test; ++p) {
                Xadj[p] = X[p] - beta_cv * (Y[p] - bs);
            }
            price = mean_of(Xadj);

            // stderr from adjusted samples
            double var = 0.0;
            for (int p = 0; p < n_test; ++p) {
                const double d = Xadj[p] - price;
                var += d * d;
            }
            var /= static_cast<double>(std::max(n_test - 1, 1));

            LSMPriceResult res;
            res.price = price;
            res.mc_stderr = std::sqrt(var / static_cast<double>(n_test));
            return res;
        }
    }

    // stderr without CV
    double var = 0.0;
    for (int p = 0; p < n_test; ++p) {
        const double d = X[p] - price;
        var += d * d;
    }
    var /= static_cast<double>(std::max(n_test - 1, 1));

    LSMPriceResult res;
    res.price = price;
    res.mc_stderr = std::sqrt(var / static_cast<double>(n_test));
    return res;
}

} // namespace lsm
