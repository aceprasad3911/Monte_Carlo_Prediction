# Monte Carlo Portfolio Backtester & VaR Diagnostics

![Project Banner](docs/banner.png)

## Overview
This project implements a **rolling Monte Carlo backtesting framework** for multi-asset portfolios, with **formal risk measures** and **diagnostics visualizations**. It’s designed for **quant research, portfolio analysis, and reproducible finance simulations**.

With this tool, you can:
- Simulate portfolio returns under various **weighting strategies** (Equal Weight, Minimum Variance, Max Sharpe, Risk Parity)
- Forecast **Value-at-Risk (VaR)** and **Conditional VaR (CVaR)**
- Conduct formal **VaR backtesting** (Kupiec POF test, Christoffersen independence, Basel traffic-light classification)
- Generate **diagnostic plots**: innovation distributions, forecast volatility, portfolio performance, VaR breaches

This project is **Python-based**, fully reproducible, and suitable as a **portfolio showcase**.

---

## Features

### 1. Rolling Monte Carlo Backtesting
- Flexible **window** and **horizon** settings
- Supports **Normal, Student-t, and Bootstrap innovations**
- Multiple **volatility models**: Constant, EWMA, GARCH
- Simulates **portfolio-level returns** across all assets

### 2. Risk Metrics & Backtesting
- Computes **VaR & CVaR** per simulation
- Performs **Kupiec POF** and **Christoffersen independence** tests
- Basel traffic-light classification for risk assessment
- Saves all statistics as **CSV files** for reproducibility

### 3. Diagnostics & Visualization
- Plots **representative innovations**
- Plots **forecast volatility**
- Portfolio performance & VaR breach visualizations
- Automatically saves figures to structured directories

---

## Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/Monte_Carlo_Prediction.git
cd Monte_Carlo_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
---

## Usage

```bash
from src.backtest import backtest_with_plots
import pandas as pd

# Load historical price data
prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)

# Run backtest
results = backtest_with_plots(
    prices=prices,
    window=252,
    horizon=5,
    n_sims=1000,
    weight_fn=equal_weight,
    innovation_type="normal",
    vol_type="garch",
    alpha=0.05,
    diagnostics=True
)
```
---
## Project Structure

```bash
Monte_Carlo_Prediction/
│
├─ src/
│  ├─ backtest.py                # Main backtesting functions
│  ├─ risk.py                    # VaR & CVaR calculations
│  ├─ innovations.py             # Innovation generators
│  ├─ volatility.py              # Volatility models (Constant, EWMA, GARCH)
│  ├─ diagnostics/               # Diagnostics data collection
│  │  └─ diagnostics_plots.py    # Innovation & volatility plots
│  ├─ plots/                     # Backtest performance plots
│  │  └─ backtest_plots.py
│  └─ utils.py                   # Directory & helper functions
│
├─ experiments/
│  └─ run_simulation.py          # Example backtest runs
│
├─ data/                         # Price datasets
├─ figures/                      # Auto-saved plots
├─ requirements.txt              # Python dependencies
└─ README.md                     # Project overview

```
---
## Dependencies

Python ≥3.9

numpy, pandas, scipy, matplotlib

Optional: arch (for GARCH volatility)

---
## Future Work

Integrate interactive Streamlit dashboard for real-time portfolio simulations

Add portfolio optimization visualizations

Extend to multi-horizon risk scenarios