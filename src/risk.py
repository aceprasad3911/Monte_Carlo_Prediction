#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/risk.py

import numpy as np
import pandas as pd
from scipy.stats import chi2
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


def kupiec_pof_test(
    hits: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Kupiec (1995) Proportion of Failures test.

    hits: Boolean array where True = no breach, False = VaR breach
    """
    T = len(hits)
    x = np.sum(~hits)  # number of VaR breaches

    if x == 0:
        return {
            "LR_pof": 0.0,
            "p_value": 1.0,
            "breaches": 0,
            "expected": alpha * T
        }

    p_hat = x / T

    LR_pof = -2 * (
        (T - x) * np.log((1 - alpha) / (1 - p_hat))
        + x * np.log(alpha / p_hat)
    )

    p_value = 1 - chi2.cdf(LR_pof, df=1)

    return {
        "LR_pof": float(LR_pof),
        "p_value": float(p_value),
        "breaches": int(x),
        "expected": float(alpha * T)
    }


def christoffersen_independence_test(
    hits: np.ndarray
) -> Dict[str, float]:
    """
    Christoffersen (1998) independence test.
    """
    hits = (~hits).astype(int)  # 1 = breach

    n00 = n01 = n10 = n11 = 0

    for t in range(1, len(hits)):
        if hits[t-1] == 0 and hits[t] == 0:
            n00 += 1
        elif hits[t-1] == 0 and hits[t] == 1:
            n01 += 1
        elif hits[t-1] == 1 and hits[t] == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0

    L_ind = (
        n00 * safe_log(1 - pi0) +
        n01 * safe_log(pi0) +
        n10 * safe_log(1 - pi1) +
        n11 * safe_log(pi1)
    )

    L_uncond = (
        (n00 + n10) * safe_log(1 - pi) +
        (n01 + n11) * safe_log(pi)
    )

    LR_ind = -2 * (L_uncond - L_ind)
    p_value = 1 - chi2.cdf(LR_ind, df=1)

    return {
        "LR_ind": float(LR_ind),
        "p_value": float(p_value)
    }


def basel_traffic_light(
    breaches: int,
    T: int
) -> str:
    """
    Basel traffic light classification (for 5% VaR, ~250 obs).
    """
    if breaches <= 4:
        return "Green"
    elif breaches <= 9:
        return "Yellow"
    else:
        return "Red"
