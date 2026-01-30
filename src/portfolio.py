# src/portfolio.py


import numpy as np


def portfolio_paths(
    asset_paths: np.ndarray,
    weights: np.ndarray
) -> np.ndarray:
    """
    Aggregate asset-level paths into portfolio paths.

    Parameters
    ----------
    asset_paths : np.ndarray
        Shape (T, sims, N)
    weights : np.ndarray
        Shape (N,)

    Returns
    -------
    np.ndarray
        Portfolio paths with shape (T, sims)
    """
    return asset_paths @ weights


# ---------------- Portfolio Construction Rules ---------------- #

def equal_weight(n_assets: int) -> np.ndarray:
    return np.ones(n_assets) / n_assets


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ np.ones(len(cov))
    return w / w.sum()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Long-only, fully-invested max Sharpe portfolio.
    """
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    w = np.clip(w, 0, None)
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()
