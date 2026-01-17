# LSM-Delta-Hedge

A **quant-desk style** project that implements a **C++ Longstaff–Schwartz (LSM) Bermudan put pricer** (Eigen) exposed to Python via **pybind11**, and a **historical delta-hedging simulator** on **SPY**.

The goal is to reproduce the workflow of an options market maker / hedging desk:
1) price an American-style option on a discrete exercise grid (Bermudan put),
2) compute model delta,
3) run a historical replay and delta-hedge the short option position,
4) measure hedging error, turnover, and transaction-cost sensitivity,
5) validate correctness against analytic benchmarks and convergence behavior,
6) demonstrate desk-grade engineering (stateful model objects, CI, tests, reproducible runs).

---

## What this project does

We simulate the **hedged book** for a **short Bermudan put** on SPY:

- **Historical replay:** daily SPY closes (Yahoo Finance).
- **Volatility input:** rolling realized volatility (no look-ahead).
- **Rates:** risk-free rate (constant by default; can be extended).
- **Pricing & delta:** C++ LSM engine under risk-neutral GBM.
- **Daily hedging:** re-balance to model delta, accrue cash at r, apply transaction costs.
- **Risk & performance metrics:** book value, PnL, max drawdown, VaR/CVaR, turnover, total transaction cost.
- **Reproducibility:** single-command experiments that save CSV + JSON + plots into `results/`.

---

## Why LSM + Bermudan puts

- **Early exercise** is meaningful for puts (especially in high-rate environments).
- Longstaff–Schwartz is a standard Monte Carlo method for **American-style pricing** on a discrete exercise grid.
- Bermudan (discrete exercise) is **more realistic to implement and test** than a continuous-time American formulation while capturing the core challenge.

---

## Methods (overview)

### 1) Market data (no look-ahead)
We compute features using only information available at time *t*:

- **Close prices** from Yahoo Finance.
- **Log returns:** `log(S_t / S_{t-1})`.
- **Realized vol:** rolling window standard deviation of log returns, annualized:
- 
  $$
  \sigma_t = \mathrm{std}(\text{logret}) \sqrt{252}
  $$
  
  shifted by one day to avoid look-ahead.

### 2) Risk-neutral model and LSM pricing
We assume GBM under Q:

$$
\frac{dS_t}{S_t} = (r-q)\,dt + \sigma\,dW_t
$$


LSM algorithm:
- Simulate Monte Carlo paths.
- Work backwards over exercise dates.
- Regress discounted continuation values on basis functions of state (default: **Laguerre** basis).
- Exercise if intrinsic value exceeds estimated continuation.

Key engineering choices:
- **Out-of-sample regression**: split paths into train/test (reduces in-sample bias).
- **Regularization**: ridge regression solved via augmented QR (stability).
- **Control variate** (optional): European put payoff vs Black–Scholes to reduce variance.

### 3) Delta estimation (CRN finite differences + frozen regression)
Delta uses central differences with common random numbers (CRN):

$$
\Delta \approx \frac{C(S+\epsilon)-C(S-\epsilon)}{2\epsilon}, \quad \epsilon = 10^{-4}S
$$


We avoid retraining regressions for bumps:
- Train LSM regression once (betas frozen),
- Re-apply the policy to bumped paths (CRN).

### 4) Hedging simulator (book accounting)
We track the marked hedged book:

$$
V_t = \text{Cash}_t + \Delta_t S_t - C_t
$$


Cash accrues at continuous-compounded rate (daily):

$$
\text{Cash}_{t+\Delta t} = \text{Cash}_t e^{r\Delta t} - (\Delta_t-\Delta_{t-1})S_t - \text{TC}_t
$$


Transaction costs:
- linear model in **bps of traded notional**:
  
  $$
  \text{TC}_t = \frac{\text{bps}}{10^4}\,|\Delta_t-\Delta_{t-1}|\,S_t
  $$
  

Early exercise (pragmatic heuristic):
- Bermudan/American value ≥ intrinsic; exercise region approximated when **time value ~ 0**:

  $$
  C_t - \text{payoff}_t \approx 0
  $$
  
  plus a delta gate (deep ITM put).

---

## Desk-dev upgrade: stateful model object (fast hedging)
A key realism upgrade is the **stateful C++ LSMModel**:
- Train once (store regression betas + test paths).
- Quote daily without retraining using a `start_step` offset.
- Python `LSMModelManager` rebuilds weekly or on vol/rate thresholds.

This mirrors a desk setup:
- **recalibrate on regimes**, not every tick/day.

---

## Repository structure

- `src/cpp/lsm/`
  - `lsm_model.*`: stateless pricing + delta (baseline implementation)
  - `lsm_model_object.*`: **stateful LSMModel** (desk-dev upgrade)
- `src/cpp/bindings/`
  - `pybind_module.cpp`: pybind11 module (`lsm_cpp`) exposing functions + class
- `python/lsmhedge/`
  - `data.py`: data + features (no look-ahead)
  - `hedge.py`: historical hedging loop
  - `model_cache.py`: stateful model manager + recalibration policy
  - `metrics.py`: drawdown, VaR/CVaR, Sharpe, turnover
  - `validators.py`: sanity checks for model risk controls
  - `bs.py`: Black–Scholes reference (tests)
- `tests/`
  - C++ smoke tests + Python tests (including BS benchmark test)
- `notebooks/`
  - `01_backtest.py`: backtest/plots (script-style notebook)
- `python/run_experiment.py`
  - **single-command reproducibility harness** that writes results to `results/`
- `python/studies/`
  - `multi_start_study.py`: run many start dates and summarize distributions
- `python/bench/`
  - `stateful_vs_stateless.py`: speed comparison

---

## Quickstart

### 1) Dependencies
- CMake + C++17 compiler
- Python 3.10+ recommended
- Python packages:
```bash
pip install numpy pandas matplotlib yfinance pytest pybind11
````

### 2) Build the extension

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3) Run the backtest (basic)

```bash
python notebooks/01_backtest.py
```

---

## Reproduce results (one command)

From repo root:

```bash
python python/run_experiment.py --stateful --save-plots --do-sweep
```

This writes a timestamped folder under `results/` containing:

* `baseline_path.csv`
  Daily path with spot, delta, hedge trades, cash, book value, PnL, etc.
* `baseline_report.json`
  Summary metrics (final PnL, ann vol, drawdown, VaR/CVaR, turnover, costs).
* `baseline_validators.json`
  Sanity checks (price ≥ intrinsic, delta bounds, etc.).
* `sweep_summary.csv`
  Small grid over transaction costs and vol windows (and optionally policies).
* `plots/*.png` (if enabled)
  Spot vs hedge, book value/PnL, PnL histogram.

---

## Multi-start regime study (distribution, not cherry-picked)

```bash
python python/studies/multi_start_study.py
```

Outputs `results/regime_<timestamp>/regime_summary.csv`, which summarizes outcomes across many start dates (roughly monthly), useful for showing hedging performance stability across regimes.

---

## Speed comparison (desk-dev story)

```bash
python python/bench/stateful_vs_stateless.py
```

Prints runtime and speedup of:

* stateless daily retraining vs
* stateful weekly recalibration

---

## Testing and CI

* C++ smoke tests validate basic sanity and delta behavior.
* Python tests include:

  * European (steps=1) LSM vs Black–Scholes put check
  * hedging loop smoke tests
  * stateful model smoke test
* CI builds the extension and runs tests.

Run locally:

```bash
PYTHONPATH=python pytest -q
```

---

## Notes on realism / limitations

* Uses GBM with realized-vol input; real markets have volatility clustering/jumps.
* Delta hedging is discrete (daily) and uses a simple TC model.
* Early exercise is approximated via a time-value heuristic (no direct continuation value exposed yet).
* Still, the project is designed to quantify **hedging error and model risk** in a controlled, reproducible way.


