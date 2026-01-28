# src/backtest.py

import numpy as np
import pandas as pd
from typing import Callable
from pathlib import Path

from .gbm import monte_carlo_multivariate_paths
from .portfolio import portfolio_paths
from .risk import var_cvar, var_backtest
from .config import TRADING_DAYS
from .plots import plot_backtest_performance, plot_var_hits
from .utils import get_figures_dir


def rolling_backtest(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable
) -> pd.DataFrame:
    """
    Rolling Monte Carlo VaR backtest.
    """

    results = []

    for t in range(window, len(prices) - horizon):

        train = prices.iloc[t - window : t]
        test = prices.iloc[t : t + horizon + 1]

        # Estimate GBM parameters
        train_rets = np.log(train / train.shift(1)).dropna()
        mu = train_rets.mean().values * TRADING_DAYS
        cov = train_rets.cov().values * TRADING_DAYS

        # Portfolio weights
        weights = weight_fn(mu, cov)

        # Monte Carlo simulation
        paths = monte_carlo_multivariate_paths(
            S0=train.iloc[-1].values,
            mu=mu,
            cov=cov,
            days=horizon,
            sims=n_sims
        )

        portfolio_mc = portfolio_paths(paths, weights)

        # Realised return
        realised = test.values @ weights
        realised_return = realised[-1] / realised[0] - 1

        # Forecast distribution
        forecast_returns = portfolio_mc[-1] / portfolio_mc[0] - 1
        var, cvar = var_cvar(forecast_returns)

        results.append({
            "date": prices.index[t],
            "realised_return": realised_return,
            "forecast_mean": forecast_returns.mean(),
            "forecast_VaR": var,
            "forecast_CVaR": cvar,
            "hit_VaR": realised_return >= var
        })

    return pd.DataFrame(results)


def backtest_with_plots(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable,
    alpha: float = 0.05,
    model_name: str = "gbm",
    performance_file: str | None = None,
    var_file: str | None = None,
    stats_file: str | None = None
) -> pd.DataFrame:
    """
    Run rolling backtest + VaR statistical validation.
    """

    results = rolling_backtest(
        prices=prices,
        window=window,
        horizon=horizon,
        n_sims=n_sims,
        weight_fn=weight_fn
    )

    figures_dir: Path = get_figures_dir()

    # ------------------ Plots ------------------

    plot_backtest_performance(
        results,
        title="Rolling Backtest: Realised Return vs Forecast VaR",
        filename=figures_dir / (performance_file or f"backtest_performance_{model_name}.png")
    )

    plot_var_hits(
        results,
        title="VaR Breaches Over Time",
        filename=figures_dir / (var_file or f"var_breaches_{model_name}.png")
    )

    # ------------------ VaR Backtest Statistics ------------------

    stats_df = var_backtest(
        realized_returns=results["realised_return"],
        var_series=results["forecast_VaR"],
        alpha=alpha
    )

    stats_path = figures_dir / (stats_file or f"var_stats_{model_name}.csv")
    stats_df.to_csv(stats_path, index=False)

    # Console output (variable: value format)
    print(f"\n=== VaR Backtest Statistics ({model_name.upper()}) ===")
    for col in stats_df.columns:
        print(f"{col:20s}: {stats_df.iloc[0][col]}")

    return results
