import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable

TRADING_DAYS = 252
DT = 1 / TRADING_DAYS


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
    dt = DT

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
    w = inv_cov @ np.ones(len(cov))  # One asset dominates → huge leverage → terrible drawdowns due to unconstrained optimisation.
    return w / w.sum()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Long-only, no-leverage constrained Max Sharpe.
    """
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    w = np.clip(w, 0, None)  # Long-only constraint
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


def max_drawdown_paths(paths: np.ndarray) -> float:
    """
    Compute mean maximum drawdown across Monte Carlo paths.
    """
    drawdowns = []
    for i in range(paths.shape[1]):
        p = paths[:, i]
        peak = np.maximum.accumulate(p)
        dd = (p - peak) / peak
        drawdowns.append(dd.min())
    return float(np.mean(drawdowns))


#  ----------------- 8) Simulation Summary  ---------------------------------------

def summarise(paths: np.ndarray) -> Dict[str, float]:
    final_vals = paths[-1]
    returns = final_vals / paths[0] - 1

    var, cvar = var_cvar(returns)
    mdd = max_drawdown_paths(paths)

    return {
        "mean_final": float(final_vals.mean()),
        "median_final": float(np.median(final_vals)),
        "VaR_5%": float(var),
        "CVaR_5%": float(cvar),
        "max_drawdown": float(mdd)
    }


#  ----------------- 9) Rolling Backtest  ---------------------------------------

def rolling_backtest(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable
) -> pd.DataFrame:

    results = []

    for t in range(window, len(prices) - horizon):

        # -----------------------------
        # 1. Split data
        # -----------------------------
        train_prices = prices.iloc[t-window:t]
        test_prices = prices.iloc[t:t+horizon+1]

        train_rets = np.log(train_prices / train_prices.shift(1)).dropna()

        # -----------------------------
        # 2. Estimate parameters (daily)
        # -----------------------------
        mu = train_rets.mean().values * TRADING_DAYS
        cov = train_rets.cov().values * TRADING_DAYS

        # -----------------------------
        # 3. Portfolio construction
        # -----------------------------
        weights = weight_fn(mu, cov)

        # -----------------------------
        # 4. Monte Carlo simulation
        # -----------------------------
        paths = monte_carlo_multivariate_paths(
            S0=train_prices.iloc[-1].values,
            mu=mu,
            cov=cov,
            days=horizon,
            sims=n_sims
        )

        portfolio_mc = portfolio_paths(paths, weights)

        # -----------------------------
        # 5. Realised portfolio path
        # -----------------------------
        realised = test_prices.values @ weights
        realised_return = realised[-1] / realised[0] - 1

        # -----------------------------
        # 6. Forecast distribution
        # -----------------------------
        forecast_returns = portfolio_mc[-1] / portfolio_mc[0] - 1
        var, cvar = var_cvar(forecast_returns)

        # -----------------------------
        # 7. Store results
        # -----------------------------
        results.append({
            "date": prices.index[t],
            "realised_return": realised_return,
            "forecast_mean": forecast_returns.mean(),
            "forecast_VaR": var,
            "forecast_CVaR": cvar,
            "hit_VaR": realised_return >= var
        })

    return pd.DataFrame(results)


#  ----------------- 9) Main Driver  ---------------------------------------

def main():

    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    start, end = "2020-01-01", "2025-01-01"
    days, sims = 252, 10_000

    prices = get_price_matrix(tickers, start, end)
    mu, cov, _ = estimate_gbm_params(prices)

    S0 = prices.iloc[-1].values

    asset_paths = monte_carlo_multivariate_paths(
        S0, mu, cov, days=days, sims=sims
    )

    strategies = {
        "Equal Weight": equal_weight(len(tickers)),
        "Min Variance": min_variance_weights(cov),
        "Max Sharpe": max_sharpe_weights(mu, cov),
        "Risk Parity": risk_parity_weights(cov),
    }

    for name, w in strategies.items():
        port_paths = portfolio_paths(asset_paths, w)
        stats = summarise(port_paths)

        print(f"\n=== {name} ===")
        for k, v in stats.items():
            print(f"{k:15s}: {v:.4f}")


if __name__ == "__main__":
    main()



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
    """
    Summarise Monte Carlo portfolio paths with
    distribution-based and path-dependent risk metrics.
    """
    final_vals = paths[-1]
    returns = final_vals / paths[0] - 1

    var, cvar = var_cvar(returns)
    mdd = max_drawdown_paths(paths)

    return {
        "mean_final": float(final_vals.mean()),
        "median_final": float(np.median(final_vals)),
        "VaR_5%": float(var),
        "CVaR_5%": float(cvar),
        "max_drawdown": float(mdd)
    }

def rolling_backtest(
    prices: pd.DataFrame,
    window: int,
    horizon: int,
    n_sims: int,
    weight_fn: Callable
) -> pd.DataFrame:

    results = []

    for t in range(window, len(prices) - horizon):

        # -----------------------------
        # 1. Split data
        # -----------------------------
        train_prices = prices.iloc[t-window:t]
        test_prices  = prices.iloc[t:t+horizon+1]

        train_rets = np.log(train_prices / train_prices.shift(1)).dropna()
        test_rets  = np.log(test_prices / test_prices.shift(1)).dropna()

        # -----------------------------
        # 2. Estimate parameters
        # -----------------------------
        mu  = train_rets.mean().values
        cov = train_rets.cov().values

        # -----------------------------
        # 3. Portfolio construction
        # -----------------------------
        weights = weight_fn(mu, cov)

        # -----------------------------
        # 4. Monte Carlo simulation
        # -----------------------------
        paths = simulate_gbm(
            S0=train_prices.iloc[-1].values,
            mu=mu,
            cov=cov,
            T=horizon * DT,
            steps=horizon,
            n_sims=n_sims
        )

        portfolio_paths = aggregate_portfolio(paths, weights)

        # -----------------------------
        # 5. Realised portfolio path
        # -----------------------------
        realised_prices = test_prices.values @ weights
        realised_returns = realised_prices / realised_prices[0] - 1

        # -----------------------------
        # 6. Forecast distribution
        # -----------------------------
        forecast_final = portfolio_paths[-1]
        forecast_var, forecast_cvar = var_cvar(
            forecast_final / portfolio_paths[0] - 1
        )

        # -----------------------------
        # 7. Store results
        # -----------------------------
        results.append({
            "date": prices.index[t],
            "realised_return": realised_returns[-1],
            "forecast_mean": forecast_final.mean() / portfolio_paths[0, 0] - 1,
            "forecast_VaR": forecast_var,
            "forecast_CVaR": forecast_cvar,
            "hit_VaR": realised_returns[-1] >= forecast_var
        })

    return pd.DataFrame(results)


#  ----------------- 9) Main Driver  ---------------------------------------

def main():

    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    start, end = "2020-01-01", "2025-01-01"
    days, sims = 252, 10_000

    prices = get_price_matrix(tickers, start, end)
    S0 = prices.iloc[-1].values

    mu, cov, _ = estimate_gbm_params(prices)

    asset_paths = monte_carlo_multivariate_paths(
        S0, mu, cov, days=days, sims=sims
    )

    strategies = {
        "Equal Weight": equal_weight(len(tickers)),
        "Min Variance": min_variance_weights(cov),
        "Max Sharpe": max_sharpe_weights(mu, cov),
        "Risk Parity": risk_parity_weights(cov),
    }

    for name, w in strategies.items():
        port_paths = portfolio_paths(asset_paths, w)

        final_vals = port_paths[-1]
        rets = final_vals / port_paths[0] - 1

        var, cvar = var_cvar(rets)
        mdd = max_drawdown_paths(port_paths)

        print(f"\n=== {name} ===")
        print(f"mean_final     : {final_vals.mean():.4f}")
        print(f"median_final   : {np.median(final_vals):.4f}")
        print(f"VaR_5%         : {var:.4f}")
        print(f"CVaR_5%        : {cvar:.4f}")
        print(f"max_drawdown   : {mdd:.4f}")


if __name__ == "__main__":
    main()
