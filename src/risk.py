#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/risk.py

import numpy as np
from typing import Dict


def var_cvar(returns: np.ndarray, alpha: float = 0.05):
    var = np.percentile(returns, alpha * 100)
    cvar = returns[returns <= var].mean()
    return var, cvar


def max_drawdown_paths(paths: np.ndarray) -> float:
    """
    Mean max drawdown across Monte Carlo paths.
    """
    drawdowns = []
    for i in range(paths.shape[1]):
        p = paths[:, i]
        peak = np.maximum.accumulate(p)
        dd = (p - peak) / peak
        drawdowns.append(dd.min())
    return float(np.mean(drawdowns))


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
