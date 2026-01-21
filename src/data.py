#  /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/data.py


import yfinance as yf
import pandas as pd
from typing import List


# single-asset models represented as vector, portfolios represented as matrix P ∈ ℝ^(T × N)
def get_price_matrix(
    tickers: List[str],
    start: str,
    end: str
) -> pd.DataFrame:

    """
    Download adjusted close prices for a multiple tickers.
    Returns a DataFrame indexed by date with one column per asset.
    """

    if len(tickers) < 2:
        raise ValueError("Provide at least two tickers.")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # Adjusts for splits & dividends
        progress=False
    )["Close"]

    if data.empty:
        raise ValueError("No price data returned.")

    # At this point:
    #   type(data) == pd.DataFrame
    #   data.shape == (T, N)
    return data.dropna(how="all")  # Drop rows where all assets are missing
