"""Tests that verify the properties that matter in a backtester.

The two most valuable tests here — and worth mentioning in your README /
interviews — are the lookahead test and the null test:

  * test_no_lookahead: a strategy that "knows" tomorrow's return should NOT
    be profitable after the engine's execution lag. If this test fails, the
    engine leaks future information.
  * test_null_data_no_edge: on pure random walks, momentum should produce a
    deflated Sharpe near zero. A pipeline that finds edges in noise is worse
    than no pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from quantbt.data import generate_synthetic
from quantbt.engine import Backtest, CostModel
from quantbt.metrics import sharpe_ratio, max_drawdown, performance_summary
from quantbt.strategy import Strategy, Momentum, MeanReversion
from quantbt.validation import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    walk_forward,
    block_bootstrap,
)


class SameDayOracle(Strategy):
    """Cheating strategy: its signal on day t is the sign of day t's OWN
    return. That information only exists at day t's close, so trading on it
    at day t's close price is lookahead. A correct engine lags execution by
    one day, making this signal worthless on a random walk; a broken engine
    (no lag) would let it earn |r| every single day."""

    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        todays_ret = prices.pct_change()
        return np.sign(todays_ret).fillna(0.0)


class BuyAndHold(Strategy):
    def compute_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
        return w


def test_no_lookahead():
    prices = generate_synthetic(n_assets=1, n_days=2000, regime_switching=False)
    strat = SameDayOracle()
    res = Backtest(prices, strat, CostModel(0, 0)).run()
    sr_lagged = sharpe_ratio(res["net_return"])

    # Prove the test has teeth: WITHOUT the lag this oracle earns |r| daily.
    rets = prices.pct_change().fillna(0.0)
    unlagged = (strat.compute_weights(prices) * rets).sum(axis=1)
    sr_unlagged = sharpe_ratio(unlagged)

    assert sr_unlagged > 5.0, "Oracle should be wildly profitable without the lag"
    assert abs(sr_lagged) < 1.0, f"Lookahead bias detected: oracle Sharpe = {sr_lagged:.2f}"


def test_costs_reduce_returns():
    prices = generate_synthetic(n_days=1500)
    free = Backtest(prices, Momentum(lookback=60), CostModel(0, 0)).run()
    costly = Backtest(prices, Momentum(lookback=60), CostModel(5, 10)).run()
    assert costly["equity"].iloc[-1] < free["equity"].iloc[-1]
    assert (costly["cost"] >= 0).all()


def test_costs_proportional_to_turnover():
    prices = generate_synthetic(n_days=1000)
    cm = CostModel(commission_bps=2, slippage_bps=3)
    res = Backtest(prices, MeanReversion(lookback=10, entry_z=1.0), cm).run()
    expected = res["turnover"] * cm.total_bps / 10_000
    assert np.allclose(res["cost"], expected)


def test_buy_and_hold_matches_underlying():
    prices = generate_synthetic(n_assets=1, n_days=1000)
    res = Backtest(prices, BuyAndHold(), CostModel(0, 0)).run()
    strat_total = res["equity"].iloc[-1] / res["equity"].iloc[0]
    # held weight becomes 1.0 on day index 1, capturing the day-0 -> day-1
    # return onward, i.e. the full price path.
    asset_total = prices.iloc[-1, 0] / prices.iloc[0, 0]
    assert abs(strat_total / asset_total - 1) < 1e-9


def test_nan_prices_rejected():
    prices = generate_synthetic(n_days=100)
    prices.iloc[5, 0] = np.nan
    with pytest.raises(ValueError):
        Backtest(prices, Momentum())


def test_psr_more_data_more_confidence():
    rng = np.random.default_rng(1)
    daily_edge = 0.0004
    short = pd.Series(daily_edge + 0.01 * rng.standard_normal(200))
    long = pd.Series(daily_edge + 0.01 * rng.standard_normal(5000))
    assert probabilistic_sharpe_ratio(long) > probabilistic_sharpe_ratio(short)


def test_dsr_penalizes_many_trials():
    rng = np.random.default_rng(2)
    rets = pd.Series(0.0005 + 0.01 * rng.standard_normal(1000))
    few = deflated_sharpe_ratio(rets, n_trials=2, trials_sr_var=0.001)
    many = deflated_sharpe_ratio(rets, n_trials=500, trials_sr_var=0.001)
    assert many["dsr"] < few["dsr"]


def test_null_data_no_edge():
    """The most important integration test: on random walks the full
    walk-forward + DSR pipeline must NOT certify an edge."""
    prices = generate_synthetic(n_assets=3, n_days=2200, trend_strength=0.0, seed=7)
    wf = walk_forward(
        prices, Momentum, {"lookback": [21, 63, 126], "long_only": [True]},
        train_days=504, test_days=126,
    )
    result = deflated_sharpe_ratio(
        wf["oos_returns"], n_trials=wf["n_trials"],
        all_trial_sharpes=wf["all_trial_sharpes"],
    )
    assert result["dsr"] < 0.95, "Pipeline certified an edge on pure noise!"


def test_trending_data_momentum_wins():
    """Positive control: with genuine autocorrelation, momentum should beat
    its performance on noise."""
    trend = generate_synthetic(n_assets=3, n_days=2000, trend_strength=0.08, seed=3)
    noise = generate_synthetic(n_assets=3, n_days=2000, trend_strength=0.0, seed=3)
    sr_trend = sharpe_ratio(Backtest(trend, Momentum(lookback=21)).run()["net_return"])
    sr_noise = sharpe_ratio(Backtest(noise, Momentum(lookback=21)).run()["net_return"])
    assert sr_trend > sr_noise


def test_block_bootstrap_shapes():
    prices = generate_synthetic(n_days=800)
    res = Backtest(prices, Momentum(lookback=42)).run()
    boot = block_bootstrap(res["net_return"], n_sims=200)
    assert {"sharpe", "max_drawdown"} <= set(boot.columns)
    assert boot.loc["5%", "max_drawdown"] <= boot.loc["95%", "max_drawdown"]


def test_performance_summary_keys():
    prices = generate_synthetic(n_days=600)
    res = Backtest(prices, Momentum(lookback=42)).run()
    s = performance_summary(res)
    assert "Sharpe Ratio" in s.index and "Max Drawdown" in s.index
    assert s["Max Drawdown"] <= 0
