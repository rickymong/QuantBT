"""Data loading and generation.

Three sources:
  1. load_yfinance  — pull daily bars for equities/futures/crypto tickers
  2. load_csv       — load your own data (any OHLCV csv)
  3. generate_synthetic — regime-switching GBM for testing the engine
                          without a network connection (and for validating
                          that the engine finds *nothing* on random data,
                          which is an important sanity check).

All loaders return a pandas DataFrame of CLOSE prices indexed by date,
one column per asset. The engine is deliberately close-to-close: intraday
fills are a common source of silent lookahead bias in student projects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_yfinance(tickers: list[str], start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Download adjusted daily closes from Yahoo Finance.

    Works for equities ("SPY", "AAPL"), continuous futures ("ES=F", "CL=F"),
    and crypto ("BTC-USD", "ETH-USD"). Requires `pip install yfinance`.

    Note on futures: Yahoo's continuous contracts are *not* back-adjusted,
    so returns across roll dates contain roll gaps. Fine for a first pass;
    see README for how to do it properly with a roll-adjusted series.
    """
    import yfinance as yf  # imported lazily so the rest of the package works offline

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"]
    if isinstance(closes, pd.Series):  # single ticker
        closes = closes.to_frame(tickers[0])
    return closes.dropna(how="all").ffill()


def load_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load a wide CSV of close prices (date column + one column per asset)."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    return df.astype(float)


def generate_synthetic(
    n_assets: int = 3,
    n_days: int = 2500,
    seed: int = 42,
    regime_switching: bool = True,
    trend_strength: float = 0.0,
) -> pd.DataFrame:
    """Simulate daily close prices.

    With trend_strength=0 the series are (regime-switching) random walks —
    a strategy backtested on them SHOULD show no significant edge. If your
    validation pipeline reports a high deflated Sharpe on this data, the
    pipeline is broken. Use this as a null-hypothesis test of your own code.

    With trend_strength>0, returns get mild positive autocorrelation, so
    momentum strategies should genuinely work — a positive control.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n_days)

    prices = {}
    for i in range(n_assets):
        # two volatility regimes (calm / stressed), Markov switching
        if regime_switching:
            vol = np.empty(n_days)
            state = 0  # 0=calm, 1=stressed
            for t in range(n_days):
                if state == 0 and rng.random() < 0.02:
                    state = 1
                elif state == 1 and rng.random() < 0.10:
                    state = 0
                vol[t] = 0.010 if state == 0 else 0.028
        else:
            vol = np.full(n_days, 0.015)

        eps = rng.standard_normal(n_days)
        rets = np.empty(n_days)
        rets[0] = vol[0] * eps[0]
        for t in range(1, n_days):
            # AR(1) in returns when trend_strength > 0
            rets[t] = trend_strength * rets[t - 1] + vol[t] * eps[t]

        prices[f"ASSET_{i+1}"] = 100 * np.exp(np.cumsum(rets))

    return pd.DataFrame(prices, index=dates)
