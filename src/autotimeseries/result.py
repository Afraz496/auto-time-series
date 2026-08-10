"""Forecast result containers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """Point forecasts and prediction intervals in tidy, labelled form."""

    mean: pd.Series
    lower: dict[float, pd.Series]
    upper: dict[float, pd.Series]
    model_name: str

    def to_frame(self) -> pd.DataFrame:
        """Return point forecasts and all intervals as a DataFrame."""
        columns: dict[str, pd.Series] = {"mean": self.mean}
        for level in sorted(self.lower):
            label = f"{level:g}"
            columns[f"lower_{label}"] = self.lower[level]
            columns[f"upper_{label}"] = self.upper[level]
        return pd.DataFrame(columns, index=self.mean.index)

    def interval(self, level: float = 95) -> pd.DataFrame:
        """Return the lower and upper bounds for one coverage level."""
        if level not in self.lower:
            raise KeyError(f"Level {level} was not computed; available: {sorted(self.lower)}")
        return pd.DataFrame({"lower": self.lower[level], "upper": self.upper[level]})
