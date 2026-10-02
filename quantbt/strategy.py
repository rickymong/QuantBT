"""Strategy interface.

A Strategy maps a price history to *target weights* per asset, one row per
day. The engine — not the strategy — is responsible for lagging signals so
that today's weight is computed only from information available *yesterday*.
Keeping that lag in exactly one place (engine.py) is how this codebase
avoids lookahead bias structurally rather than by hoping every strategy
author remembers to .shift(1).

Weights are interpreted as fractions of portfolio equity. 1.0 = 100% long,
-1.0 = 100% short. Rows may sum to more than 1 (leverage) — the engine
does not stop you, but the cost model will charge you for the turnover.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Subclass this and implement compute_weights().

    The `params` dict is the single source of truth for tunable parameters —
    the walk-forward optimizer varies strategies purely through `params`,
    so anything you might want to tune belongs there.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of target weights, same index/columns as prices.

        IMPORTANT: row t may use prices up to and including row t.
        The engine applies the one-day execution lag for you.
        """

    def __repr__(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{type(self).__name__}({p})"


class Momentum(Strategy):
    """Cross-sectional + time-series momentum.

    Long assets whose trailing `lookback`-day return is positive, short those
    where it is negative (if `long_only=False`), equal risk budget per asset.
    """

    def __init__(self, lookback: int = 126, long_only: bool = True):
        super().__init__(lookback=lookback, long_only=long_only)

    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        lb = self.params["lookback"]
        mom = prices.pct_change(lb)
        signal = np.sign(mom)
        if self.params["long_only"]:
            signal = signal.clip(lower=0)
        # equal-weight across assets with an active signal
        n_active = signal.abs().sum(axis=1).replace(0, np.nan)
        weights = signal.div(n_active, axis=0).fillna(0.0)
        return weights


class MeanReversion(Strategy):
    """Z-score mean reversion: fade moves beyond `entry_z` standard deviations
    from the `lookback`-day mean; flat inside the band."""

    def __init__(self, lookback: int = 20, entry_z: float = 1.5):
        super().__init__(lookback=lookback, entry_z=entry_z)

    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        lb, entry = self.params["lookback"], self.params["entry_z"]
        ma = prices.rolling(lb).mean()
        sd = prices.rolling(lb).std()
        z = (prices - ma) / sd
        signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        signal[z > entry] = -1.0   # stretched high -> short
        signal[z < -entry] = 1.0   # stretched low  -> long
        n_active = signal.abs().sum(axis=1).replace(0, np.nan)
        return signal.div(n_active, axis=0).fillna(0.0)


class VolTargetedMomentum(Strategy):
    """Momentum with inverse-volatility position sizing and a portfolio-level
    volatility target. This is closer to how real CTAs size positions and is
    a nice talking point in interviews: raw sign-based momentum has lumpy,
    regime-dependent risk; vol targeting stabilizes it."""

    def __init__(self, lookback: int = 126, vol_lookback: int = 63, target_vol: float = 0.10):
        super().__init__(lookback=lookback, vol_lookback=vol_lookback, target_vol=target_vol)

    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        rets = prices.pct_change()
        mom_sign = np.sign(prices.pct_change(p["lookback"]))
        asset_vol = rets.rolling(p["vol_lookback"]).std() * np.sqrt(252)
        # inverse-vol sizing, scaled so the portfolio targets `target_vol`
        raw = mom_sign * (p["target_vol"] / asset_vol.replace(0, np.nan))
        n = prices.shape[1]
        weights = (raw / n).clip(-1.5, 1.5).fillna(0.0)
        return weights
