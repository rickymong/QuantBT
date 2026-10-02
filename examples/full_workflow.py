"""End-to-end research workflow demo.

Run:  python examples/full_workflow.py            (synthetic data, offline)
      python examples/full_workflow.py --live     (real data via yfinance)

This is the workflow to show in your README:
  1. Load data (equities + futures + crypto, or synthetic offline)
  2. Naive full-sample backtest  -> the number everyone else reports
  3. Walk-forward OOS backtest   -> the number you should report
  4. Deflated Sharpe ratio       -> is the OOS result luck, given N trials?
  5. Bootstrap confidence bands  -> how bad could it plausibly get?
  6. Parameter sensitivity       -> is the edge a plateau or a spike?
"""

import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantbt import (
    Backtest,
    CostModel,
    Momentum,
    block_bootstrap,
    deflated_sharpe_ratio,
    generate_synthetic,
    parameter_sensitivity,
    performance_summary,
    walk_forward,
)


def main(live: bool):
    # ------------------------------------------------------------- 1. data
    if live:
        from quantbt import load_yfinance
        # one from each asset class the engine supports
        prices = load_yfinance(["SPY", "ES=F", "BTC-USD"], start="2016-01-01")
        prices = prices.dropna()  # crypto trades weekends; intersect calendars
        print(f"Loaded {prices.shape[1]} assets, {len(prices)} days from Yahoo Finance")
    else:
        prices = generate_synthetic(n_assets=3, n_days=2300, trend_strength=0.05)
        print("Using synthetic data (mild trend injected as a positive control).")

    costs = CostModel(commission_bps=1, slippage_bps=4)

    # ---------------------------------------------- 2. naive full-sample run
    naive = Backtest(prices, Momentum(lookback=126), costs).run()
    print("\n=== Naive full-sample backtest (what most projects stop at) ===")
    print(performance_summary(naive).round(3).to_string())

    # ------------------------------------------------------ 3. walk-forward
    grid = {"lookback": [21, 42, 63, 126, 189, 252], "long_only": [True, False]}
    wf = walk_forward(prices, Momentum, grid, train_days=756, test_days=126, cost_model=costs)
    print("\n=== Walk-forward out-of-sample ===")
    print(f"OOS Sharpe: {wf['oos_sharpe']:.3f}")
    print(f"Windows: {len(wf['windows'])}, total parameter trials: {wf['n_trials']}")
    print(f"Overfit gap (IS - OOS Sharpe): {wf['overfit_gap']:.3f}")
    print(wf["windows"][["test_start", "chosen_params", "in_sample_sharpe",
                         "out_of_sample_sharpe"]].to_string(index=False))

    # -------------------------------------------------- 4. deflated Sharpe
    dsr = deflated_sharpe_ratio(
        wf["oos_returns"], n_trials=wf["n_trials"],
        all_trial_sharpes=wf["all_trial_sharpes"],
    )
    print("\n=== Deflated Sharpe Ratio ===")
    print(f"DSR: {dsr['dsr']:.3f}  "
          f"(luck benchmark = {dsr['luck_benchmark_sr_annualized']:.2f} ann. Sharpe "
          f"over {dsr['n_trials']} trials)")
    print(dsr["verdict"])

    # ------------------------------------------------------- 5. bootstrap
    boot = block_bootstrap(wf["oos_returns"], n_sims=1000)
    print("\n=== Block-bootstrap distribution of OOS performance ===")
    print(boot.loc[["5%", "50%", "95%"]].round(3).to_string())

    # ---------------------------------------------- 6. parameter sensitivity
    sens = parameter_sensitivity(prices, Momentum,
                                 {"lookback": [21, 42, 63, 126, 189, 252],
                                  "long_only": [True]}, costs)
    print("\n=== Parameter sensitivity (full sample, diagnostic only) ===")
    print(sens.round(3).to_string(index=False))

    # ------------------------------------------------------------- plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    naive["equity"].plot(ax=axes[0, 0], title="Naive full-sample equity")
    (1 + wf["oos_returns"]).cumprod().plot(ax=axes[0, 1],
                                           title="Walk-forward OOS equity (honest)")
    axes[1, 0].bar(sens["lookback"].astype(str), sens["sharpe"])
    axes[1, 0].set_title("Sharpe vs lookback (plateau = robust)")
    ws = wf["windows"]
    axes[1, 1].plot(ws["test_start"], ws["in_sample_sharpe"], "o-", label="in-sample")
    axes[1, 1].plot(ws["test_start"], ws["out_of_sample_sharpe"], "s-", label="out-of-sample")
    axes[1, 1].legend(); axes[1, 1].set_title("IS vs OOS Sharpe per window")
    fig.tight_layout()
    fig.savefig("workflow_report.png", dpi=120)
    print("\nSaved plots to workflow_report.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="use yfinance instead of synthetic data")
    main(ap.parse_args().live)
