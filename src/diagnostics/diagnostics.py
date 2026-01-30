import numpy as np
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class DiagnosticStore:
    innovations: list = field(default_factory=list)
    volatility: list = field(default_factory=list)
    dates: list = field(default_factory=list)

    def record(self, date, z_sample, vol_forecast):
        """
        Record a single step of innovations and forecast volatility.
        z_sample: 1D np.ndarray
        vol_forecast: 1D np.ndarray
        """
        self.dates.append(date)
        self.innovations.append(z_sample)
        self.volatility.append(vol_forecast)

    def to_dataframe(self):
        """
        Returns a DataFrame with mean volatility per date.
        """
        return pd.DataFrame({
            "date": self.dates,
            "volatility": [v.mean() for v in self.volatility]
        })