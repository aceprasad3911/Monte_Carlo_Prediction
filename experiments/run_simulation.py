# src/run_simulation.py

import numpy as np
from src.data import get_price_matrix
from src.portfolio import max_sharpe_weights
from src.backtest import backtest_with_plots


def main():

    innovation_types = ["normal", "student_t", "bootstrap"]
    vol_types = ["constant", "ewma", "garch"]

    prices = get_price_matrix(
        ["AAPL", "MSFT", "GOOG", "SPY"],
        start="2020-01-01",
        end="2025-01-01",
    )

    for innovation in innovation_types:
        for vol in vol_types:
            print(f"\nRunning: {innovation} + {vol}")

            backtest_with_plots(
                prices=prices,
                window=252,
                horizon=21,
                n_sims=3_000,
                weight_fn=max_sharpe_weights,
                innovation_type=innovation,
                vol_type=vol,
            )


if __name__ == "__main__":
    main()
