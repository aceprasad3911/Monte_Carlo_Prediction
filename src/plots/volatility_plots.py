# src/plots/volatility_plots.py


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_volatility_paths(vols, title, filename: Path):
    plt.figure(figsize=(9, 4))
    plt.plot(vols, alpha=0.7)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Volatility")
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
