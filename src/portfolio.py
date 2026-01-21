#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/portfolio.py
import numpy as np


def portfolio_paths(asset_paths: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Aggregate asset paths into portfolio paths.
    """
    return asset_paths @ weights


def equal_weight(n: int) -> np.ndarray:
    return np.ones(n) / n


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ np.ones(len(cov))
    return w / w.sum()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Long-only, no-leverage constrained Max Sharpe.
    """
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ mu
    w = np.clip(w, 0, None)
    return w / w.sum()


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    inv_vol = 1 / vol
    return inv_vol / inv_vol.sum()
