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


#  ----------------- 7) Main Driver  ---------------------------------------
def main():
    tickers = ["AAPL", "MSFT", "GOOG", "SPY"]
    start = "2020-01-01"
    end = "2025-01-01"
    days = 252
    sims = 10_000

    print(f"\nDownloading data for {tickers}...")
    prices = get_price_matrix(tickers, start, end)

    S0 = prices.iloc[-1].values
    print("Initial prices:", dict(zip(tickers, S0.round(2))))

    mu, cov, log_rets = estimate_gbm_params(prices)

    print("\nAnnualised expected returns:")
    for t, m in zip(tickers, mu):
        print(f"{t}: {m:.2%}")

    print("\nSimulating correlated GBM paths...")
    asset_paths = monte_carlo_paths_multivariate(
        S0, mu, cov, days=days, sims=sims
    )

    # Equal-weight portfolio
    weights = np.ones(len(tickers)) / len(tickers)
    portfolio = portfolio_paths(asset_paths, weights)

    stats = summarise_simulation(portfolio)

    print("\n== Portfolio simulation summary ==")
    for k, v in stats.items():
        print(f"{k:12s}: {v:.2f}")

    # Risk-style metrics
    prob_loss_10 = (portfolio[-1] < 0.9 * portfolio[0]).mean()
    prob_gain_10 = (portfolio[-1] > 1.1 * portfolio[0]).mean()

    print(f"\nProbability of >10% loss: {prob_loss_10:.1%}")
    print(f"Probability of >10% gain: {prob_gain_10:.1%}")

    plot_portfolio_paths(portfolio)
    plot_final_distribution(portfolio)

    print("\nSaved plots:")
    print("- portfolio_paths.png")
    print("- portfolio_final_distribution.png")


if __name__ == "__main__":
    main()
