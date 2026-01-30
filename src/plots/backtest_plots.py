# src/plots/backtest_plots.py



import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_backtest_performance(
    results: pd.DataFrame,
    title: str,
    filename: Path
):
    """
    Plot realised returns vs forecast VaR.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        results["date"],
        results["realised_return"],
        label="Realised Return",
        linewidth=1.4
    )

    plt.plot(
        results["date"],
        results["forecast_VaR"],
        label="Forecast VaR (α)",
        linestyle="--"
    )

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)

    filename.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_var_hits(results, title="VaR Breaches", filename=None):
    """
    Plot realized portfolio returns vs VaR and mark breaches.
    Only considers the realized portfolio path to match stats.
    """

    dates = results["date"]
    realised = results["realised_return"]
    var = results["forecast_VaR"]

    # Breaches boolean series
    breaches = results["hit_VaR"]

    plt.figure(figsize=(12, 6))

    # Plot realized returns
    plt.plot(dates, realised, label="Realized Return", color="blue")

    # Plot VaR
    plt.plot(dates, var, label="VaR Forecast", color="red", linestyle="--")

    # Highlight breaches
    plt.scatter(
        dates[breaches],
        realised[breaches],
        color="orange",
        marker="x",
        s=80,
        label="VaR Breach",
        zorder=5
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()
