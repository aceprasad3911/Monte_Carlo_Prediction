# src/plots.py



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from .utils import get_figures_dir


def _resolve_path(filename: Optional[Union[str, Path]]) -> Optional[Path]:
    if filename is None:
        return None

    path = Path(filename)

    if not path.is_absolute() and path.parent == Path("."):
        path = get_figures_dir() / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ----------------- 1) Monte Carlo Path Diagnostics -----------------

def plot_portfolio_paths(
    paths: np.ndarray,
    title: str,
    filename: Optional[Union[str, Path]] = None,
    n_sample_paths: int = 100
):
    plt.figure(figsize=(9, 5))
    plt.plot(paths[:, :n_sample_paths], alpha=0.4)
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Portfolio Value")
    plt.grid(alpha=0.3)

    path = _resolve_path(filename)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------- 2) Terminal Distribution -----------------

def plot_final_distribution(
    paths: np.ndarray,
    title: str,
    filename: Optional[Union[str, Path]] = None,
    bins: int = 60
):
    plt.figure(figsize=(8, 5))
    plt.hist(paths[-1], bins=bins, alpha=0.75)
    plt.title(title)
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)

    path = _resolve_path(filename)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------- 3) Rolling Backtest Diagnostics -----------------

def plot_backtest_performance(
    results: pd.DataFrame,
    title: str,
    filename: Optional[Union[str, Path]] = None
):
    plt.figure(figsize=(10, 5))

    plt.plot(
        results["date"],
        results["realised_return"],
        label="Realised Return",
        linewidth=1.5
    )

    plt.plot(
        results["date"],
        results["forecast_VaR"],
        label="Forecast VaR (5%)",
        linestyle="--"
    )

    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)

    path = _resolve_path(filename)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------- 4) VaR Hit Rate Diagnostic -----------------

def plot_var_hits(
    results: pd.DataFrame,
    title: str,
    filename: Optional[Union[str, Path]] = None
):
    breaches = results[~results["hit_VaR"]]

    plt.figure(figsize=(10, 4))
    plt.plot(results["date"], results["realised_return"], label="Realised Return")
    plt.scatter(
        breaches["date"],
        breaches["realised_return"],
        color="red",
        label="VaR Breach",
        zorder=5
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(alpha=0.3)

    path = _resolve_path(filename)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
