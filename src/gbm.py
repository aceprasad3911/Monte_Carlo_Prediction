#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/gbm.py


import numpy as np
import pandas as pd
from typing import Tuple
from .config import TRADING_DAYS, DT

def estimate_gbm_params(
    prices: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

    """
    Estimate annualised drift (mu) and volatility (sigma)
    from historical log-returns.
    """

    # Compute log returns (element-wise) -> implements r_t,i = ln(P_t,i / P_{t-1,i}) -> Vectorised across all assets.
    log_rets = np.log(prices / prices.shift(1)).dropna()

    # Estimate parameters w/t Daily → annual via: μ_annual  = 252 × mean_daily & Σ_annual  = 252 × cov_daily
    mu = log_rets.mean().values * TRADING_DAYS
    cov = log_rets.cov().values * TRADING_DAYS

    return mu, cov, log_rets


def monte_carlo_multivariate_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    days: int,
    sims: int,
    seed: int = 42
) -> np.ndarray:

    """
    Simulate correlated GBM paths for multiple assets.

    Returns array of shape:
    (days + 1, sims, n_assets)
    """

    np.random.seed(seed)

    n_assets = len(S0)
    L = np.linalg.cholesky(cov)

    Z = np.random.normal(size=(days, sims, n_assets))
    correlated_Z = Z @ L.T

    drift = (mu - 0.5 * np.diag(cov)) * DT

    log_paths = np.zeros((days + 1, sims, n_assets))
    log_paths[0] = np.log(S0)

    for t in range(1, days + 1):
        log_paths[t] = (
            log_paths[t-1]
            + drift
            + np.sqrt(DT) * correlated_Z[t - 1]
        )

    return np.exp(log_paths)
