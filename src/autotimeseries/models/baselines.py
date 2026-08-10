"""Fast, robust benchmark forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Random-walk forecast using the most recent observation."""

    def _fit(self, y: pd.Series) -> None:
        self.last_ = float(y.iloc[-1])

    def _fitted_values(self) -> np.ndarray:
        return np.r_[np.nan, self.y_.to_numpy()[:-1]]

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        steps = np.arange(1, horizon + 1)
        return np.repeat(self.last_, horizon), np.sqrt(self.sigma2_ * steps)


class MeanForecaster(BaseForecaster):
    """Forecast the historical mean."""

    def _fit(self, y: pd.Series) -> None:
        self.mean_ = float(y.mean())

    def _fitted_values(self) -> np.ndarray:
        return np.repeat(self.mean_, len(self.y_))

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        se = np.sqrt(self.sigma2_ * (1 + 1 / len(self.y_)))
        return np.repeat(self.mean_, horizon), np.repeat(se, horizon)


class DriftForecaster(BaseForecaster):
    """Random walk with drift between the first and last observations."""

    def _fit(self, y: pd.Series) -> None:
        if len(y) < 2:
            raise ValueError("DriftForecaster requires at least two observations")
        self.drift_ = float((y.iloc[-1] - y.iloc[0]) / (len(y) - 1))

    def _fitted_values(self) -> np.ndarray:
        return np.r_[np.nan, self.y_.to_numpy()[:-1] + self.drift_]

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        steps = np.arange(1, horizon + 1)
        mean = float(self.y_.iloc[-1]) + self.drift_ * steps
        se = np.sqrt(self.sigma2_ * steps * (1 + steps / max(len(self.y_) - 1, 1)))
        return mean, se


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeat values from the latest seasonal cycle."""

    def __init__(self, seasonal_period: int):
        if not isinstance(seasonal_period, int) or seasonal_period < 1:
            raise ValueError("seasonal_period must be a positive integer")
        self.seasonal_period = seasonal_period

    def _fit(self, y: pd.Series) -> None:
        if len(y) <= self.seasonal_period:
            raise ValueError("y must contain more than one seasonal cycle boundary")
        self.season_ = y.iloc[-self.seasonal_period :].to_numpy()

    def _fitted_values(self) -> np.ndarray:
        return np.r_[
            np.full(self.seasonal_period, np.nan), self.y_.to_numpy()[: -self.seasonal_period]
        ]

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(horizon) % self.seasonal_period
        cycles = np.floor(np.arange(horizon) / self.seasonal_period) + 1
        return self.season_[idx], np.sqrt(self.sigma2_ * cycles)
