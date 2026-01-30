import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd


def plot_innovation_distribution(z, title, filename):
    """
    Plot the histogram of standardized innovations with normal overlay.
    z: 1D np.ndarray
    """
    plt.figure(figsize=(8,5))
    plt.hist(z, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    x = np.linspace(z.min(), z.max(), 500)
    plt.plot(x, stats.norm.pdf(x), lw=2, color='red', label="Standard Normal")
    plt.title(title)
    plt.xlabel("Innovation")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_volatility_series(df: pd.DataFrame, title, filename):
    """
    Plot the mean forecast volatility over time.
    df: DataFrame with 'date' and 'volatility' columns
    """
    plt.figure(figsize=(10,5))
    plt.plot(df["date"], df["volatility"], color='blue', lw=2)
    plt.title(title)
    plt.ylabel("Mean Forecast Volatility")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()