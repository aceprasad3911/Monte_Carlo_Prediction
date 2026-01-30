# src/plots/innovation_plots.py


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_innovation_distribution(Z, title, filename: Path):
    plt.figure(figsize=(7, 4))
    plt.hist(Z.flatten(), bins=80, alpha=0.75)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
