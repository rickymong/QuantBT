"""Performance metrics.

All ratio metrics are annualized assuming daily data (252 periods/year)
unless stated otherwise. Every function takes a Series of *net* returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe. Note: assumes iid returns, which real returns
    violate (autocorrelation biases this up for smooth strategies) —
    one more reason the validation module exists."""
    excess = returns - rf_annual / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return np.nan
    return excess.mean() / sd * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    excess = returns - rf_annual / TRADING_DAYS
    downside = excess[excess < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return excess.mean() / downside * np.sqrt(TRADING_DAYS)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline of the equity curve (negative number)."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return ((equity - peak) / peak).min()


def calmar_ratio(returns: pd.Series) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if len(active) == 0:
        return np.nan
    return (active > 0).mean()


def performance_summary(result: pd.DataFrame) -> pd.Series:
    """Summary stats from a Backtest.run() result frame."""
    r = result["net_return"]
    return pd.Series(
        {
            "Annualized Return": annualized_return(r),
            "Annualized Vol": annualized_vol(r),
            "Sharpe Ratio": sharpe_ratio(r),
            "Sortino Ratio": sortino_ratio(r),
            "Max Drawdown": max_drawdown(r),
            "Calmar Ratio": calmar_ratio(r),
            "Hit Rate": hit_rate(r),
            "Avg Daily Turnover": result["turnover"].mean(),
            "Total Cost Drag (ann.)": result["cost"].mean() * TRADING_DAYS,
            "Final Equity": result["equity"].iloc[-1],
        }
    )
