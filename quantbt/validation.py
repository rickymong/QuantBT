"""Statistical validation — the part that makes a backtest believable.

Core problem: if you try N strategy variants and report the best one, the
best in-sample Sharpe ratio is inflated by selection bias even if every
variant is pure noise. This module implements the standard defenses:

  * probabilistic_sharpe_ratio — P(true Sharpe > benchmark) accounting for
    sample length, skew and fat tails of the return series.
  * deflated_sharpe_ratio — PSR with the benchmark raised to the expected
    maximum Sharpe among N independent noise trials (Bailey & López de
    Prado, 2014, "The Deflated Sharpe Ratio"). This is the headline number:
    DSR < 0.95 means you cannot reject the hypothesis that your "best"
    strategy is the luckiest of your N attempts.
  * walk_forward — rolling train/test parameter selection, so reported
    performance is genuinely out-of-sample.
  * block_bootstrap — stationary block bootstrap confidence intervals for
    Sharpe and max drawdown, preserving autocorrelation structure.
  * parameter_sensitivity — an edge that exists only at lookback=37 and
    vanishes at 35 and 39 is an artifact, not an edge.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from quantbt.engine import Backtest, CostModel
from quantbt.metrics import sharpe_ratio, max_drawdown

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe ratio inference
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """P(true Sharpe > benchmark_sr), following Bailey & López de Prado.

    Works with the NON-annualized (per-period) Sharpe and benchmark, and
    corrects the standard error for skewness and kurtosis of returns —
    fat-tailed, negatively skewed strategies (e.g. short-vol) need much
    more data to prove the same edge.
    """
    r = returns.dropna()
    n = len(r)
    if n < 30:
        return np.nan
    sr = r.mean() / r.std()  # per-period
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # normal = 3
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, trials_sr_var: float) -> float:
    """Expected maximum per-period Sharpe among n_trials independent
    zero-skill strategies whose SR estimates have variance trials_sr_var."""
    if n_trials < 2:
        return 0.0
    return np.sqrt(trials_sr_var) * (
        (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
        + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe_ratio(
    best_returns: pd.Series,
    n_trials: int,
    trials_sr_var: float | None = None,
    all_trial_sharpes: list[float] | None = None,
) -> dict:
    """Deflated Sharpe Ratio of the best strategy out of n_trials attempts.

    Pass either `all_trial_sharpes` (per-period Sharpes of every variant you
    tried — preferred) or an explicit `trials_sr_var`.

    Returns a dict with the DSR, the luck benchmark it was measured against,
    and a verdict string. Interpretation: DSR is P(true SR > expected max SR
    of N lucky noise strategies). Conventionally you want DSR >= 0.95.
    """
    if all_trial_sharpes is not None:
        trials_sr_var = float(np.var(all_trial_sharpes, ddof=1))
    if trials_sr_var is None:
        raise ValueError("Provide trials_sr_var or all_trial_sharpes.")

    sr_benchmark = expected_max_sharpe(n_trials, trials_sr_var)
    dsr = probabilistic_sharpe_ratio(best_returns, benchmark_sr=sr_benchmark)
    return {
        "dsr": dsr,
        "n_trials": n_trials,
        "luck_benchmark_sr_annualized": sr_benchmark * np.sqrt(252),
        "verdict": (
            "PASS: performance unlikely to be selection luck (DSR >= 0.95)"
            if dsr >= 0.95
            else "FAIL: cannot distinguish from the best of N noise strategies"
        ),
    }


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

def walk_forward(
    prices: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list],
    train_days: int = 756,   # ~3 years
    test_days: int = 126,    # ~6 months
    cost_model: CostModel | None = None,
    selection_metric=sharpe_ratio,
) -> dict:
    """Rolling walk-forward optimization.

    For each window: fit (grid-search) params on `train_days`, then apply the
    winning params to the following `test_days`. Concatenated test segments
    form the out-of-sample return series — the only performance you should
    ever quote.

    Also returns every in-sample trial Sharpe, which is exactly the input the
    deflated Sharpe ratio needs (it must know how many things you tried).
    """
    cost_model = cost_model or CostModel()
    keys = list(param_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*param_grid.values())]

    oos_returns = []
    window_log = []
    all_trial_sharpes: list[float] = []

    start = 0
    while start + train_days + test_days <= len(prices):
        train = prices.iloc[start : start + train_days]
        test = prices.iloc[start + train_days - 260 : start + train_days + test_days]
        # test slice includes a lookback buffer so indicators warm up before
        # the OOS period starts; only the true OOS rows are kept below.

        best_params, best_score = None, -np.inf
        for params in combos:
            res = Backtest(train, strategy_cls(**params), cost_model).run()
            score = selection_metric(res["net_return"])
            all_trial_sharpes.append(
                res["net_return"].mean() / res["net_return"].std()
                if res["net_return"].std() > 0 else 0.0
            )
            if not np.isnan(score) and score > best_score:
                best_params, best_score = params, score

        oos_res = Backtest(test, strategy_cls(**best_params), cost_model).run()
        oos_slice = oos_res["net_return"].iloc[-test_days:]
        oos_returns.append(oos_slice)
        window_log.append(
            {
                "train_start": train.index[0],
                "test_start": oos_slice.index[0],
                "test_end": oos_slice.index[-1],
                "chosen_params": best_params,
                "in_sample_sharpe": best_score,
                "out_of_sample_sharpe": sharpe_ratio(oos_slice),
            }
        )
        start += test_days

    oos = pd.concat(oos_returns)
    log = pd.DataFrame(window_log)
    return {
        "oos_returns": oos,
        "oos_sharpe": sharpe_ratio(oos),
        "windows": log,
        "n_trials": len(combos) * len(log),
        "all_trial_sharpes": all_trial_sharpes,
        # in-sample minus out-of-sample Sharpe: large positive gap = overfitting
        "overfit_gap": float((log["in_sample_sharpe"] - log["out_of_sample_sharpe"]).mean()),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def block_bootstrap(
    returns: pd.Series,
    n_sims: int = 2000,
    avg_block: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Stationary block bootstrap (Politis & Romano) of the return series.

    Resamples returns in random-length blocks (geometric, mean `avg_block`)
    to preserve short-range autocorrelation and volatility clustering, then
    reports the distribution of annualized Sharpe and max drawdown.
    Quote the 5th-percentile Sharpe, not the point estimate.
    """
    rng = np.random.default_rng(seed)
    r = returns.dropna().to_numpy()
    n = len(r)
    p = 1.0 / avg_block

    sharpes, mdds = np.empty(n_sims), np.empty(n_sims)
    for s in range(n_sims):
        idx = np.empty(n, dtype=int)
        t = rng.integers(0, n)
        for i in range(n):
            idx[i] = t
            if rng.random() < p:
                t = rng.integers(0, n)
            else:
                t = (t + 1) % n
        sim = pd.Series(r[idx])
        sharpes[s] = sharpe_ratio(sim)
        mdds[s] = max_drawdown(sim)

    return pd.DataFrame({"sharpe": sharpes, "max_drawdown": mdds}).describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )


# ---------------------------------------------------------------------------
# Parameter sensitivity
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    prices: pd.DataFrame,
    strategy_cls,
    param_grid: dict[str, list],
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Full-sample Sharpe for every parameter combination.

    Not a substitute for walk-forward — this is diagnostic. A robust edge
    shows a smooth plateau of similar Sharpes across neighboring parameters;
    a single spike surrounded by noise is curve-fitting.
    """
    cost_model = cost_model or CostModel()
    keys = list(param_grid)
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        res = Backtest(prices, strategy_cls(**params), cost_model).run()
        rows.append({**params, "sharpe": sharpe_ratio(res["net_return"])})
    return pd.DataFrame(rows)
