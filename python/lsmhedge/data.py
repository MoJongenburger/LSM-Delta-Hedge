from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    yf = None
    _yf_import_error = e


@dataclass(frozen=True)
class MarketDataConfig:
    symbol: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2024-12-31"

    # underlying price column
    price_col: str = "Close"

    # realized vol
    vol_window: int = 60
    trading_days: int = 252

    # risk-free rate
    rf_mode: Literal["constant", "yahoo"] = "constant"
    rf_constant: float = 0.04
    rf_yahoo_symbol: str = "^IRX"  # 13w T-bill (Yahoo); reported as % p.a. typically

    # dividend yield (optional)
    dividend_yield_mode: Literal["none", "trailing_1y"] = "none"


def _require_yfinance() -> None:
    if yf is None:
        raise ImportError(
            "yfinance is required for market data download. Install via `pip install yfinance`.\n"
            f"Original error: {_yf_import_error}"
        )


def download_yahoo_ohlc(symbol: str, start: str, end: str) -> pd.DataFrame:
    _require_yfinance()
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No OHLC data returned for {symbol}. Check symbol/date range.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def download_yahoo_rate_irx(start: str, end: str, symbol: str = "^IRX") -> pd.Series:
    """
    Downloads a short-rate proxy like ^IRX.
    Yahoo typically reports ^IRX close as a percent (e.g., 5.25).
    We convert to a continuous-compounded annual rate: r = log(1 + y_decimal).
    """
    _require_yfinance()
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No rate data returned for {symbol}.")
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index)
    y_decimal = (s / 100.0).clip(lower=0.0)     # 0.0525
    r_cont = np.log1p(y_decimal)                # ~0.0512
    r_cont.name = "r"
    return r_cont


def estimate_dividend_trailing_sum(symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    """
    Returns dividends aligned to index (daily series) and trailing 1y sum.
    """
    _require_yfinance()
    tkr = yf.Ticker(symbol)
    div = tkr.dividends
    if div is None or div.empty:
        out = pd.Series(0.0, index=index, name="div_trailing_1y")
        return out

    div = div.copy()
    div.index = pd.to_datetime(div.index)

    div_daily = div.reindex(index).fillna(0.0)
    trailing = div_daily.rolling(252).sum()
    trailing.name = "div_trailing_1y"
    return trailing


def prepare_market_data(cfg: MarketDataConfig) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with columns:
      close, logret, sigma, r, q

    No look-ahead:
      - sigma uses returns up to t-1 via shift(1)
      - r is shifted(1) if sourced
      - q is shifted(1) if computed
    """
    df = download_yahoo_ohlc(cfg.symbol, cfg.start, cfg.end)
    if cfg.price_col not in df.columns:
        raise ValueError(f"price_col='{cfg.price_col}' not found. Available: {list(df.columns)}")

    out = pd.DataFrame(index=df.index)
    out["close"] = df[cfg.price_col].astype(float)

    # log returns
    out["logret"] = np.log(out["close"] / out["close"].shift(1))

    # realized vol (annualized), no lookahead
    roll = out["logret"].rolling(cfg.vol_window).std()
    expd = out["logret"].expanding(min_periods=cfg.vol_window).std()
    # Use rolling when available, else expanding; both shifted to avoid look-ahead
    sigma_raw = roll.combine_first(expd)
    out["sigma"] = (sigma_raw.shift(1) * np.sqrt(cfg.trading_days))

    # risk-free r
    if cfg.rf_mode == "constant":
        out["r"] = float(cfg.rf_constant)
    elif cfg.rf_mode == "yahoo":
        r_series = download_yahoo_rate_irx(cfg.start, cfg.end, cfg.rf_yahoo_symbol)
        out["r"] = r_series.reindex(out.index).ffill().shift(1)
    else:
        raise ValueError(f"Unknown rf_mode: {cfg.rf_mode}")

    # dividend yield q
    if cfg.dividend_yield_mode == "none":
        out["q"] = 0.0
    elif cfg.dividend_yield_mode == "trailing_1y":
        trailing_div = estimate_dividend_trailing_sum(cfg.symbol, out.index)
        q_simple = (trailing_div / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        out["q"] = np.log1p(q_simple).shift(1)  # cont-comp approx
        out["q"] = out["q"].clip(lower=0.0, upper=0.20)
    else:
        raise ValueError(f"Unknown dividend_yield_mode: {cfg.dividend_yield_mode}")

    # clean: require close/sigma/r/q
    out = out.dropna(subset=["close", "sigma", "r", "q"]).copy()
    return out

