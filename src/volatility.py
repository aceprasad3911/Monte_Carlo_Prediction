# src/volatility.py


import numpy as np
import pandas as pd
from arch import arch_model


def constant_volatility(train_returns):
    """
    Constant volatility + constant correlation (GBM baseline).
    """
    vol = train_returns.std().values
    corr = train_returns.corr().values
    return vol, corr


def ewma_volatility(train_returns, lambda_=0.94):
    """
    RiskMetrics EWMA volatility and correlation.
    """
    r = train_returns.values
    cov = np.cov(r, rowvar=False)

    for t in range(len(r)):
        x = r[t][:, None]
        cov = lambda_ * cov + (1 - lambda_) * (x @ x.T)

    vol = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vol, vol)

    return vol, corr


def garch_volatility(returns: pd.DataFrame, horizon: int | None):
    """
    GARCH(1,1) volatility.

    horizon = None  -> in-sample conditional volatility (for bootstrap)
    horizon >= 1    -> out-of-sample forecast (for simulation)
    """
    vols = []

    for col in returns.columns:
        model = arch_model(
            returns[col] * 100,
            vol="Garch",
            p=1,
            q=1,
            dist="normal"
        )
        res = model.fit(disp="off")

        if horizon is None:
            # In-sample conditional volatility
            vol = res.conditional_volatility.values / 100
        else:
            # Out-of-sample forecast
            forecast = res.forecast(horizon=horizon)
            vol = np.sqrt(forecast.variance.values[-1]) / 100

        vols.append(vol)

    vols = np.column_stack(vols)

    # Correlation always estimated in-sample
    corr = returns.corr().values

    return vols, corr
