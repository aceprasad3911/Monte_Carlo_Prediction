# src/data.py


import yfinance as yf
import pandas as pd
from typing import List


def get_price_matrix(
    tickers: List[str],
    start: str,
    end: str
) -> pd.DataFrame:
    """
    Download adjusted close prices for multiple assets.

    Returns
    -------
    pd.DataFrame
        Price matrix with shape (T, N)
    """

    if len(tickers) < 2:
        raise ValueError("Provide at least two tickers.")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if "Close" not in data:
        raise ValueError("No 'Close' prices returned.")

    prices = data["Close"]

    if prices.empty:
        raise ValueError("No price data returned.")

    # Drop rows where all assets are missing
    prices = prices.dropna(how="all")

    return prices
