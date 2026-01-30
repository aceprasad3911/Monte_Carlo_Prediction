# src/innovations.py


import numpy as np
import pandas as pd


def normal_innovations(days, sims, n_assets, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(size=(days, sims, n_assets))


def student_t_innovations(days, sims, n_assets, df=5, seed=None):
    """
    Heavy-tailed innovations with unit variance.
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_t(df, size=(days, sims, n_assets))
    return Z * np.sqrt((df - 2) / df)


def bootstrap_innovations(train_returns: pd.DataFrame, days, sims, seed=None):
    """
    Historical resampling (model-free benchmark).
    """
    if seed is not None:
        np.random.seed(seed)

    hist = train_returns.values
    T, n_assets = hist.shape

    idx = np.random.randint(0, T, size=(days, sims))
    Z = hist[idx]

    return Z.reshape(days, sims, n_assets)
