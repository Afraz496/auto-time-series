"""Fast, robust benchmark forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Random-walk forecast using the most recent observation.

    Examples
    --------
    >>> import pandas as pd
    >>> from autotimeseries import NaiveForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = NaiveForecaster().fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.0, 14.0]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - The floor baseline every other model must beat; no assumptions
           about the series beyond "tomorrow looks like today"
       * - Avoid when
         - The series has a visible trend or seasonal cycle -- it will
           systematically lag both
       * - Handles trend
         - No
       * - Handles seasonality
         - No
       * - Extra dependencies
         - None
       * - Min. observations
         - 1
    """

    def _fit(self, y: pd.Series) -> None:
        self.last_ = float(y.iloc[-1])

    def _fitted_values(self) -> np.ndarray:
        return np.r_[np.nan, self.y_.to_numpy()[:-1]]

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        steps = np.arange(1, horizon + 1)
        return np.repeat(self.last_, horizon), np.sqrt(self.sigma2_ * steps)


class MeanForecaster(BaseForecaster):
    """Forecast the historical mean.

    Examples
    --------
    >>> import pandas as pd
    >>> from autotimeseries import MeanForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = MeanForecaster().fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [12.5, 12.5]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - A stability sanity check -- does the series have a signal worth
           modeling at all?
       * - Avoid when
         - The series has any trend or seasonality; it should almost
           always lose to a trend-aware model on a proper backtest
       * - Handles trend
         - No
       * - Handles seasonality
         - No
       * - Extra dependencies
         - None
       * - Min. observations
         - 1
    """

    def _fit(self, y: pd.Series) -> None:
        self.mean_ = float(y.mean())

    def _fitted_values(self) -> np.ndarray:
        return np.repeat(self.mean_, len(self.y_))

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        se = np.sqrt(self.sigma2_ * (1 + 1 / len(self.y_)))
        return np.repeat(self.mean_, horizon), np.repeat(se, horizon)


class DriftForecaster(BaseForecaster):
    """Random walk with drift between the first and last observations.

    Examples
    --------
    >>> import pandas as pd
    >>> from autotimeseries import DriftForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = DriftForecaster().fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.8, 15.6]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - A trending series with no seasonality; a much stronger baseline
           than :class:`NaiveForecaster` in that case, at the same cost
       * - Avoid when
         - The series has seasonality a straight line can't capture, or
           the trend is not roughly linear end-to-end
       * - Handles trend
         - Yes (linear, extrapolated from the first and last observation)
       * - Handles seasonality
         - No
       * - Extra dependencies
         - None
       * - Min. observations
         - 2
    """

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
    """Repeat values from the latest seasonal cycle.

    Examples
    --------
    >>> import pandas as pd
    >>> from autotimeseries import SeasonalNaiveForecaster
    >>> y = pd.Series([10.0, 20.0, 15.0, 25.0, 11.0, 21.0, 16.0, 26.0])
    >>> model = SeasonalNaiveForecaster(seasonal_period=4).fit(y)
    >>> model.predict(horizon=4).mean.round(2).tolist()
    [11.0, 21.0, 16.0, 26.0]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - The baseline to beat whenever a series has real seasonal
           structure; captures the seasonal swing for free
       * - Avoid when
         - The series has no repeating cycle, or a trend on top of the
           cycle that repeating last year's values would miss
       * - Handles trend
         - No
       * - Handles seasonality
         - Yes (repeats the last full cycle)
       * - Extra dependencies
         - None
       * - Min. observations
         - ``seasonal_period + 1``
    """

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
