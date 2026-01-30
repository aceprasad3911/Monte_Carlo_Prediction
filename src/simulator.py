# src/simulator.py


import numpy as np
from .config import DT


def simulate_paths(
    S0,
    mu,
    innovations,
    vol_type,
    base_corr,
    vol_forecast=None
):
    """
    Unified Monte Carlo engine.

    Innovations: Z_t
    Volatility: sigma_t
    Correlation: R
    """
    days, sims, n_assets = innovations.shape
    paths = np.zeros((days + 1, sims, n_assets))
    paths[0] = S0

    for t in range(days):

        if vol_type == "constant":
            sigma = vol_forecast
        elif vol_type == "ewma":
            sigma = vol_forecast
        elif vol_type == "garch":
            sigma = vol_forecast[t]
        else:
            raise ValueError("Invalid volatility model.")

        cov_t = np.diag(sigma) @ base_corr @ np.diag(sigma)
        L = np.linalg.cholesky(cov_t)

        shocks = innovations[t] @ L.T
        drift = (mu - 0.5 * sigma**2) * DT

        paths[t + 1] = paths[t] * np.exp(drift + np.sqrt(DT) * shocks)

    return paths
