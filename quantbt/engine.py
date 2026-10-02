"""Backtest engine.

Design decisions worth being able to defend in an interview:

1. EXECUTION LAG. Weights computed from data through day t are applied to
   returns on day t+1 (`weights.shift(1)`). This single line is the
   difference between a legitimate backtest and lookahead bias. It lives
   here, once, rather than inside every strategy.

2. COSTS SCALE WITH TURNOVER. Each day the portfolio pays
   (commission + slippage) * |Δweight| summed across assets. A strategy
   that trades daily gets charged accordingly; buy-and-hold pays almost
   nothing. Many naive backtesters apply a flat fee per trade or ignore
   costs entirely — high-turnover "edges" usually die here.

3. VECTORIZED, CLOSE-TO-CLOSE. Daily-bar granularity with next-close fills.
   This understates slippage for large orders and can't model limit-order
   fills, which is stated honestly as a limitation rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbt.strategy import Strategy


@dataclass
class CostModel:
    """All values in basis points of traded notional, per side.

    Rough defaults for a small retail account:
      US large-cap equities: commission ~0-1 bps, slippage ~2-5 bps
      Liquid futures (ES):   commission ~0.5 bps, slippage ~1-2 bps
      Crypto (major pairs):  commission ~5-10 bps, slippage ~5-20 bps
    """

    commission_bps: float = 1.0
    slippage_bps: float = 3.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


class Backtest:
    def __init__(
        self,
        prices: pd.DataFrame,
        strategy: Strategy,
        cost_model: CostModel | None = None,
        initial_capital: float = 100_000.0,
    ):
        if prices.isna().any().any():
            raise ValueError(
                "Price data contains NaNs. Clean or forward-fill deliberately "
                "before backtesting — silent ffill inside an engine hides data problems."
            )
        self.prices = prices
        self.strategy = strategy
        self.cost_model = cost_model or CostModel()
        self.initial_capital = initial_capital

    def run(self) -> pd.DataFrame:
        """Returns a DataFrame with columns:
        gross_return, cost, net_return, turnover, equity
        """
        asset_returns = self.prices.pct_change().fillna(0.0)

        target = self.strategy.compute_weights(self.prices)
        if not target.index.equals(self.prices.index) or list(target.columns) != list(self.prices.columns):
            raise ValueError("Strategy must return weights with the same index/columns as prices.")

        # THE lag: signals from day t trade at day t+1's close.
        held = target.shift(1).fillna(0.0)

        gross = (held * asset_returns).sum(axis=1)
        turnover = held.diff().abs().sum(axis=1).fillna(0.0)
        cost = turnover * self.cost_model.total_bps / 10_000.0
        net = gross - cost

        equity = self.initial_capital * (1 + net).cumprod()

        return pd.DataFrame(
            {
                "gross_return": gross,
                "cost": cost,
                "net_return": net,
                "turnover": turnover,
                "equity": equity,
            },
            index=self.prices.index,
        )
