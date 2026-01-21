import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

TRADING_DAYS = 252


#  --------------------------------- 1) Multi-Asset Data Loading ---------------------------------------

# single-asset models represented as vector, portfolios represented as matrix P ∈ ℝ^(T × N)
def get_price_matrix(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2025-01-01"
) -> pd.DataFrame:

    """
    Download adjusted close prices for a multiple tickers.
    Returns a DataFrame indexed by date with one column per asset.
    """

    if len(tickers) < 2:
        raise ValueError("Provide at least two tickers.")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # Adjusts for splits & dividends
        progress=False
    )["Close"]

    if isinstance(data, pd.Series):
        raise ValueError("Provide at least two tickers for a portfolio.")

        # Drop rows where all assets are missing
        data = data.dropna(how="all")

    if data.empty:
        raise ValueError("No price data returned.")

    # At this point:
    #   type(data) == pd.DataFrame
    #   data.shape == (T, N)
    return data


#  ----------------- 2) Estimate Multi-Variate GBM parameters from historical log-returns ---------------------------------------
def estimate_gbm_params(
        prices: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Estimate annualised drift (mu) and volatility (sigma)
    from historical log-returns.
    """
    # Compute log returns (element-wise) -> implements r_t,i = ln(P_t,i / P_{t-1,i}) -> Vectorised across all assets.
    log_rets = np.log(prices / prices.shift(1)).dropna(axis=0)

    # Estimate parameters w/t Daily → annual via: μ_annual  = 252 × mean_daily & Σ_annual  = 252 × cov_daily
    mu_annual = log_rets.mean().values * TRADING_DAYS
    cov_annual = log_rets.cov().values * TRADING_DAYS

    return mu_annual, cov_annual, log_rets


#  ----------------- 3) Diagnostic: Visualize Log-Returns  ---------------------------------------
def plot_log_returns(log_rets: pd.DataFrame):
    log_rets.plot(subplots=True, figsize=(10, 6), title="Log Returns")
    plt.tight_layout()
    plt.show()


#  ----------------- 4) Correlated Monte Carlo GBM simulation  ---------------------------------------
def monte_carlo_multivariate_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    days: int = 252,
    sims: int = 10_000,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate correlated GBM paths for multiple assets.

    Returns array of shape:
    (days + 1, sims, n_assets)
    """
    np.random.seed(seed)

    n_assets = len(S0)
    dt = 1 / TRADING_DAYS

    # Cholesky decomposition
    L = np.linalg.cholesky(cov)

    # random shocks
    Z = np.random.normal(size=(days, sims, n_assets))
    correlated_Z = Z @ L.T

    # Drift term
    drift = (mu - 0.5 * np.diag(cov)) * dt

    # Log-price evolution
    log_paths = np.zeros((days + 1, sims, n_assets))
    log_paths[0] = np.log(S0)

    for t in range(1, days + 1):
        log_paths[t] = (
                log_paths[t - 1]
                + drift
                + np.sqrt(dt) * correlated_Z[t - 1]
        )

    return np.exp(log_paths)


#  ----------------- 5) Portfolio Aggregation  ---------------------------------------

def portfolio_paths(
    asset_paths: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Convert asset-level paths into portfolio value paths.

    asset_paths: (days+1, sims, n_assets)
    weights:     (n_assets,)

    returns:     (days+1, sims)
    """
    return asset_paths @ weights


#  ----------------- 6) Portfolio Optimisation Strategies  ---------------------------------------

def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ np.ones(len(cov))
    return w / w.sum()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()

#  ----------------- 7) Risk Metrics  ---------------------------------------

def var_cvar(returns: np.ndarray, alpha: float = 0.05):
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar


def max_drawdown(path: np.ndarray) -> float:
    peak = np.maximum.accumulate(path)
    drawdown = (path - peak) / peak
    return drawdown.min()


#  ----------------- 8) Diagnostics & Plots  ---------------------------------------
def plot_portfolio_paths(
    paths: np.ndarray,
    title: str,
    filename: str,
    n_sample_paths: int = 100
):
    plt.figure(figsize=(8, 5))
    plt.plot(paths[:, :n_sample_paths], alpha=0.6)
    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_final_distribution(
    paths: np.ndarray,
    title: str,
    filename: str
):
    plt.figure(figsize=(8, 5))
    plt.hist(paths[-1], bins=60)
    plt.title(title)
    plt.xlabel("Final Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


#  ----------------- 7) Detailed Summary statistics  ---------------------------------------

def summarise_simulation(portfolio_paths: np.ndarray) -> dict:
    """
    Compute summary stats from simulated final prices.
    """
    final_vals = portfolio_paths[-1]
    return {
        "mean_final": float(final_vals.mean()),
        "median_final": float(np.median(final_vals)),
        "p5": float(np.percentile(final_vals, 5)),
        "p95": float(np.percentile(final_vals, 95)),
        "min": float(final_vals.min()),
        "max": float(final_vals.max())
    }


#  ----------------- 8) Simulation Summary  ---------------------------------------

def summarise(paths: np.ndarray) -> Dict[str, float]:
    final_vals = paths[-1]
    returns = final_vals / paths[0] - 1

    var, cvar = var_cvar(returns)

    return {
        "mean_final": final_vals.mean(),
        "median_final": np.median(final_vals),
        "VaR_5%": var,
        "CVaR_5%": cvar,
        "max_drawdown": max_drawdown(final_vals)
    }


#  ----------------- 9) Main Driver  ---------------------------------------

def main():

    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    prices = get_price_matrix(tickers)

    mu, cov, _ = estimate_gbm_params(prices)
    S0 = prices.iloc[-1].values

    asset_paths = monte_carlo_multivariate_paths(S0, mu, cov)

    strategies = {
        "Equal Weight": equal_weight(len(tickers)),
        "Min Variance": min_variance_weights(cov),
        "Max Sharpe": max_sharpe_weights(mu, cov),
        "Risk Parity": risk_parity_weights(cov)
    }

    for name, w in strategies.items():
        port_paths = portfolio_paths(asset_paths, w)

        stats = summarise(port_paths)

        print(f"\n=== {name} ===")
        for k, v in stats.items():
            print(f"{k:15s}: {v:.4f}")

        plot_portfolio_paths(
            port_paths,
            f"{name} – Monte Carlo Paths",
            f"{name.lower().replace(' ', '_')}_paths.png"
        )

        plot_final_distribution(
            port_paths,
            f"{name} – Final Distribution",
            f"{name.lower().replace(' ', '_')}_distribution.png"
        )

    # OPTIONAL DIAGNOSTIC
    # plot_log_returns(log_rets)


if __name__ == "__main__":
    main()
