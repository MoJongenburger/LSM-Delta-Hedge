# LSM-Delta-Hedge
This project implements a **C++ Longstaff–Schwartz (LSM) Bermudan put pricer** (Eigen) exposed to Python via **pybind11**, and a **historical delta-hedging simulator** on **SPY**.

We simulate the hedged book of a short Bermudan put:
- Daily historical SPY closes are replayed.
- Volatility is estimated via **rolling realized volatility** (no look-ahead).
- Cash accrues at the **risk-free rate**.
- The hedge is rebalanced daily using **model delta** computed via **CRN central finite differences** with **frozen regression** (train LSM once, reuse betas for bumps).
- We track **book value / PnL**, turnover, transaction costs, drawdowns, VaR/CVaR.

### Why LSM + Bermudan puts
Early exercise is relevant for puts, and LSM is a standard Monte-Carlo method for pricing/hedging American-style options on a discrete exercise grid.

## Architecture
- **C++ (`src/cpp/lsm`)**: LSM pricing engine + CRN delta (frozen regression)
- **pybind11 (`src/cpp/bindings`)**: Python bindings (GIL released during compute; C++ exceptions mapped to Python)
- **Python (`python/lsmhedge`)**: market data pipeline + hedging simulator + metrics
- **Notebook script (`notebooks/01_backtest.py`)**: reproducible backtest + plots + simple parameter sweep

## Quickstart
### 1) Get dependencies
- CMake, C++17 compiler
- Python 3.10+ recommended
- Python packages: `pip install numpy pandas matplotlib yfinance`

### 2) Add submodules
```bash
git submodule update --init --recursive
````

### 3) Build the extension

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 4) Run the backtest

```bash
python notebooks/01_backtest.py
```

## Notes on realism

* No look-ahead in sigma/r/q (shifted inputs)
* Cash accrual at continuous-compounded risk-free rate
* Transaction cost model: bps of traded notional
* Early exercise is approximated by “time value ~ 0” using model price vs intrinsic
