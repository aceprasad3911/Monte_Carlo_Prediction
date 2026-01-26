#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/experiments/run_simulation.py

import os
from src.data import get_price_matrix
from src.gbm import estimate_gbm_params, monte_carlo_multivariate_paths
from src.portfolio import (
    equal_weight,
    min_variance_weights,
    max_sharpe_weights,
    risk_parity_weights,
    portfolio_paths
)
from src.risk import summarise
from src.plots import (
    plot_portfolio_paths,
    plot_final_distribution
)
from src.backtest import backtest_with_plots  # NEW: rolling backtest + plots
from src.utils import get_figures_dir

def main():
    # 1) Load Price Data
    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    prices = get_price_matrix(tickers, start="2020-01-01", end="2025-01-01")

    # Ensure figures folder exists
    figures_dir = get_figures_dir()


    # 2) Estimate GBM Parameters
    mu, cov, _ = estimate_gbm_params(prices)
    S0 = prices.iloc[-1].values

    # 3) Monte Carlo Simulation
    asset_paths = monte_carlo_multivariate_paths(
        S0=S0,
        mu=mu,
        cov=cov,
        days=252,
        sims=10_000
    )

    # 4) Define Portfolio Strategies
    strategies = {
        "Equal Weight": equal_weight(len(tickers)),
        "Min Variance": min_variance_weights(cov),
        "Max Sharpe": max_sharpe_weights(mu, cov),
        "Risk Parity": risk_parity_weights(cov),
    }

    # 5) Run Simulations & Save Plots
    for name, w in strategies.items():
        port_paths = portfolio_paths(asset_paths, w)
        stats = summarise(portfolio_paths(asset_paths, w))

        print(f"\n=== {name} ===")
        for k, v in stats.items():
            print(f"{k:15s}: {v:.4f}")

        # Monte Carlo Diagnostic Plots
        plot_portfolio_paths(
            port_paths,
            title=f"{name} — Monte Carlo Portfolio Paths",
            filename=os.path.join(figures_dir, f"{name}_paths.png")
        )

        plot_final_distribution(
            port_paths,
            title=f"{name} — Final Value Distribution",
            filename=os.path.join(figures_dir, f"{name}_distribution.png")
        )

    # --------  Rolling Backtest (Max Sharpe)  --------
    backtest_results = backtest_with_plots(
        prices=prices,
        window=252,
        horizon=21,
        n_sims=3_000,
        weight_fn=max_sharpe_weights,
    )

    # Backtest Summary
    print("\n=== Rolling Backtest Summary (Max Sharpe) ===")
    print(backtest_results.describe())


if __name__ == "__main__":
    main()
