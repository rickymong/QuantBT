"""quantbt — a statistically rigorous backtesting engine.

Designed to answer not just "did this strategy make money in the past?"
but "is the observed performance distinguishable from luck, given how
many things I tried?"
"""

from quantbt.data import load_csv, load_yfinance, generate_synthetic
from quantbt.strategy import Strategy, Momentum, MeanReversion, VolTargetedMomentum
from quantbt.engine import Backtest, CostModel
from quantbt.metrics import performance_summary, sharpe_ratio, max_drawdown
from quantbt.validation import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    walk_forward,
    block_bootstrap,
    parameter_sensitivity,
)

__version__ = "0.1.0"
