#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/risk.py

import numpy as np
from typing import Dict

import pandas as pd
from scipy.stats import chi2


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


def kupiec_pof_test(breaches: np.ndarray, alpha: float):
    """
    Kupiec Proportion of Failures test (unconditional coverage).
    """
    T = len(breaches)
    x = breaches.sum()
    pi_hat = x / T

    if x == 0 or x == T:
        return {
            "test": "Kupiec",
            "LR_stat": np.nan,
            "p_value": 0.0,
            "reject": True
        }

    LR = -2 * (
        (T - x) * np.log((1 - alpha) / (1 - pi_hat))
        + x * np.log(alpha / pi_hat)
    )

    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "test": "Kupiec",
        "LR_stat": LR,
        "p_value": p_value,
        "reject": p_value < 0.05
    }


# Christoffersen Independence Test

def christoffersen_independence_test(breaches: np.ndarray):
    """
    Tests whether VaR breaches are independent.
    """
    b = breaches.astype(int)

    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0))
    n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0))
    n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    L0 = (
        (n00 + n10) * safe_log(1 - pi)
        + (n01 + n11) * safe_log(pi)
    )

    L1 = (
        n00 * safe_log(1 - pi01)
        + n01 * safe_log(pi01)
        + n10 * safe_log(1 - pi11)
        + n11 * safe_log(pi11)
    )

    LR = -2 * (L0 - L1)
    p_value = 1 - chi2.cdf(LR, df=1)

    return {
        "test": "Christoffersen",
        "LR_stat": LR,
        "p_value": p_value,
        "reject": p_value < 0.05
    }


# Basel Traffic Light

def basel_traffic_light(num_breaches: int, T: int, alpha: float):
    """
    Basel traffic light classification.
    """
    expected = T * alpha

    if num_breaches <= expected * 1.5:
        return "Green"
    elif num_breaches <= expected * 2.5:
        return "Yellow"
    else:
        return "Red"

def var_backtest(realized_returns, var_series, alpha=0.05):
    """
    Runs full VaR backtest suite.
    """
    breaches = (realized_returns < var_series).astype(int).values
    T = len(breaches)

    kupiec = kupiec_pof_test(breaches, alpha)
    christ = christoffersen_independence_test(breaches)
    traffic = basel_traffic_light(breaches.sum(), T, alpha)

    results = {
        "alpha": alpha,
        "n_obs": T,
        "n_breaches": breaches.sum(),
        "breach_rate": breaches.mean(),
        "kupiec_LR": kupiec["LR_stat"],
        "kupiec_p": kupiec["p_value"],
        "kupiec_reject": kupiec["reject"],
        "christ_LR": christ["LR_stat"],
        "christ_p": christ["p_value"],
        "christ_reject": christ["reject"],
        "basel_zone": traffic
    }

    return pd.DataFrame([results])
