import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple, List

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
    log_rets = np.log(prices / prices.shift(1))

    # Removes any row where at least one asset is missing.
    log_rets = log_rets.dropna(axis=0)

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
def monte_carlo_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    days: int = 252,
    sims: int = 10_000,
    seed: int = 42) -> np.ndarray:
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


#  ----------------- 5) Plotting utilities  ---------------------------------------
def plot_portfolio_paths(
    portfolio_paths: np.ndarray,
    out_file: str = "portfolio_paths.png",
    n_sample_paths: int = 100
):
    days_plus_one, sims = portfolio_paths.shape
    n = min(n_sample_paths, sims)

    plt.figure(figsize=(8, 5))
    plt.plot(portfolio_paths[:, :n], alpha=0.7, linewidth=0.8)
    plt.xlabel("Days into future")
    plt.ylabel("Portfolio value")
    plt.title("Monte Carlo Portfolio Simulation")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_final_distribution(
    portfolio_paths: np.ndarray,
    out_file: str = "portfolio_final_distribution.png"
):
    final_vals = portfolio_paths[-1]

    plt.figure(figsize=(8, 5))
    plt.hist(final_vals, bins=60)
    plt.xlabel("Final portfolio value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Portfolio Value at Horizon")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


#  ----------------- 6) Detailed Summary statistics  ---------------------------------------

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


#  ----------------- 6) Main Driver  ---------------------------------------
def main():
    # --- user input (interactive, but has defaults) ---
    ticker = input("Enter ticker [default AAPL]: ").strip().upper() or "AAPL"
    start = input("Start date [default 2020-01-01]: ").strip() or "2020-01-01"
    end = input("End date   [default 2025-01-01]: ").strip() or "2025-01-01"

    try:
        days = int(input("Forecast horizon in trading days [default 252]: ").strip() or "252")
    except ValueError:
        days = 252

    try:
        sims = int(input("Number of simulations [default 10000]: ").strip() or "10000")
    except ValueError:
        sims = 10_000

    print(f"\nDownloading data for {ticker} from {start} to {end}...")
    prices = get_price_history(ticker, start, end)
    S0 = float(prices.iloc[-1].item())
    print(f"Last observed price (S0): {S0:.2f}")

    mu, sigma, log_rets = estimate_gbm_params(prices)
    print(f"Estimated annual drift (mu):   {mu:.4f}")
    print(f"Estimated annual volatility σ: {sigma:.4f}")

    print(f"\nSimulating {sims} paths over {days} trading days...")
    paths = monte_carlo_paths(S0, mu, sigma, days=days, sims=sims)

    stats = summarise_simulation(paths)
    print("\n== Simulation summary ==")
    for k, v in stats.items():
        print(f"{k:12s}: {v:.2f}")

    # simple risk-style stats
    final_prices = paths[-1, :]
    prob_up = (final_prices > S0).mean()
    prob_down_20 = (final_prices < 0.8 * S0).mean()

    print(f"\nProbability final price > current price: {prob_up:.1%}")
    print(f"Probability final price < 80% of current: {prob_down_20:.1%}")

    # plots
    plot_price_paths(paths, ticker)
    plot_final_distribution(paths, ticker)
    print("\nSaved plots: 'price_paths.png', 'final_price_distribution.png'")


if __name__ == "__main__":
    main()