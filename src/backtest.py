# src/backtest.py

import numpy as np
import pandas as pd
from typing import Callable, Optional
from .diagnostics.diagnostics import DiagnosticStore
import warnings
from .risk import var_backtest, var_cvar
from .utils import get_figures_dir
from .plots.backtest_plots import plot_backtest_performance, plot_var_hits
from .innovations import normal_innovations, student_t_innovations
from .volatility import constant_volatility, ewma_volatility, garch_volatility
from .diagnostics.diagnostics_plots import plot_innovation_distribution, plot_volatility_series  # <-- new


# ============================================================
# Bootstrap helpers (FIX)
# ============================================================

def compute_standardized_residuals(returns: np.ndarray, vol: np.ndarray):
    eps = 1e-8
    return returns / (vol + eps)


def bootstrap_residuals(z_residuals: np.ndarray, horizon: int, n_sims: int):
    idx = np.random.randint(0, len(z_residuals), size=(horizon, n_sims))
    return z_residuals[idx]


# ============================================================
# Rolling Monte Carlo Backtest
# ============================================================

def rolling_backtest_mc(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable,
    innovation_type: str,
    vol_type: str,
    alpha: float,
    diagnostics: bool = False
) -> tuple[pd.DataFrame, Optional[DiagnosticStore]]:

    results = []
    diag_store = DiagnosticStore() if diagnostics else None

    for t in range(window, len(prices) - horizon):

        train = prices.iloc[t - window : t]
        test = prices.iloc[t : t + horizon + 1]

        # ---------------- RETURNS ----------------
        train_rets = np.log(train / train.shift(1)).dropna()
        realised_rets = np.log(test / test.shift(1)).dropna()

        mu = train_rets.mean().values
        cov = train_rets.cov().values
        weights = weight_fn(mu, cov)

        n_assets = len(mu)

        # ---------------- VOLATILITY ----------------
        if vol_type == "constant":
            vol, corr = constant_volatility(train_rets)
            vol_series = np.tile(vol, (len(train_rets), 1))
            vol_forecast = np.tile(vol, (horizon, 1))

        elif vol_type == "ewma":
            vol, corr = ewma_volatility(train_rets)
            vol_series = np.tile(vol, (len(train_rets), 1))
            vol_forecast = np.tile(vol, (horizon, 1))

        elif vol_type == "garch":
            # In-sample vol (bootstrap needs this)
            vol_series, corr = garch_volatility(train_rets, horizon=None)
            # Forecast vol (simulation needs this)
            vol_forecast, _ = garch_volatility(train_rets, horizon=horizon)

        else:
            raise ValueError(f"Unknown vol_type: {vol_type}")

        # ---------------- INNOVATIONS ----------------
        if innovation_type == "normal":
            Z = normal_innovations(horizon, n_sims, n_assets)

        elif innovation_type == "student_t":
            Z = student_t_innovations(horizon, n_sims, n_assets)

        elif innovation_type == "bootstrap":
            z_residuals = compute_standardized_residuals(
                train_rets.values, vol_series
            )
            Z = bootstrap_residuals(z_residuals, horizon, n_sims)

        else:
            raise ValueError(f"Unknown innovation_type: {innovation_type}")

        # ---------------- DIAGNOSTIC RECORD ----------------
        if diagnostics and diag_store is not None:
            diag_store.record(
                date=prices.index[t],
                z_sample=Z[0].flatten(),       # first horizon step
                vol_forecast=vol_forecast[0],  # first horizon forecast
            )

        # ---------------- SIMULATED RETURNS ----------------
        L = np.linalg.cholesky(corr)
        sim_rets = np.zeros((horizon, n_sims, n_assets))

        for h in range(horizon):
            shocks = Z[h] @ L.T
            sim_rets[h] = mu + vol_forecast[h] * shocks

        # ---------------- PORTFOLIO RETURNS ----------------
        sim_port_rets = sim_rets @ weights
        sim_port_horizon_ret = sim_port_rets.sum(axis=0)

        var, cvar = var_cvar(sim_port_horizon_ret, alpha)

        realised_port_ret = (realised_rets.values @ weights).sum()

        results.append({
            "date": prices.index[t],
            "realised_return": realised_port_ret,
            "forecast_VaR": var,
            "forecast_CVaR": cvar,
            "hit_VaR": realised_port_ret < var,
        })

    return pd.DataFrame(results), diag_store

# ============================================================
# Backtest Runner + Plots
# ============================================================


def backtest_with_plots(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable,
    innovation_type: str,
    vol_type: str,
    alpha: float = 0.05,
    diagnostics: bool = True,
) -> pd.DataFrame:

    results, diag_store = rolling_backtest_mc(
        prices=prices,
        window=window,
        horizon=horizon,
        n_sims=n_sims,
        weight_fn=weight_fn,
        innovation_type=innovation_type,
        vol_type=vol_type,
        alpha=alpha,
        diagnostics=diagnostics,
    )

    # ---------------- FIGURE DIRS ----------------
    base_dir = get_figures_dir("backtests", innovation_type, vol_type)
    stats_dir = get_figures_dir("stats")
    diag_dir = get_figures_dir("diagnostics", innovation_type, vol_type) if diagnostics else None

    # ---------------- BACKTEST PLOTS ----------------
    plot_backtest_performance(
        results,
        title=f"Backtest ({innovation_type}, {vol_type})",
        filename=base_dir / f"performance_{innovation_type}_{vol_type}.png",
    )

    plot_var_hits(
        results,
        title=f"VaR Breaches ({innovation_type}, {vol_type})",
        filename=base_dir / f"breaches_{innovation_type}_{vol_type}.png",
    )

    # ---------------- DIAGNOSTIC PLOTS ----------------
    if diagnostics and diag_store is not None:
        # Extract innovation samples
        z_all = np.concatenate(diag_store.innovations)
        plot_innovation_distribution(
            z_all,
            title=f"Representative Innovations ({innovation_type}, {vol_type})",
            filename=diag_dir / f"innovations_{innovation_type}_{vol_type}.png",
        )

        # Extract volatility series
        vol_df = diag_store.to_dataframe()
        plot_volatility_series(
            vol_df,
            title=f"Forecast Volatility ({innovation_type}, {vol_type})",
            filename=diag_dir / f"volatility_{innovation_type}_{vol_type}.png",
        )

    # ---------------- VaR STATISTICS ----------------
    stats = var_backtest(
        results["realised_return"].values,
        results["forecast_VaR"].values,
        alpha,
    )

    stats.to_csv(
        stats_dir / f"var_stats_{innovation_type}_{vol_type}.csv",
        index=False,
    )

    print(f"\n=== VaR Backtest ({innovation_type}, {vol_type}) ===")
    for col in stats.columns:
        print(f"{col:20s}: {stats.iloc[0][col]}")

    return results