"""Theta method: the first port of an R `forecast::` model into this package."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

from ..base import BaseForecaster


class ThetaForecaster(BaseForecaster):
    """Classical Theta method (Assimakopoulos & Nikolopoulos, 2000).

    Compatible Python reimplementation of R's `forecast::thetaf` (Hyndman,
    package `forecast`, GPL-3) -- not a call into R, and not yet verified
    against its numerical output; see CONTRIBUTING.md. It decomposes the
    series into two "theta lines": the theta=0 line is the long-term linear
    trend, and the theta=2 line doubles local curvature around that trend.
    The theta=2 line is extrapolated with simple exponential smoothing, the
    theta=0 line is extrapolated linearly, and the two forecasts are averaged
    with equal weight, following Assimakopoulos & Nikolopoulos (2000), "The
    theta model: a decomposition approach to forecasting", International
    Journal of Forecasting 16(4):521-530.

    When `seasonal_period` is given, the series is deseasonalized first with
    a multiplicative classical decomposition (`statsmodels.seasonal_decompose`)
    and forecasts are reseasonalized afterwards; this requires strictly
    positive values and at least two full seasonal cycles.

    Prediction intervals use the same residual-variance random-walk scaling
    (`sqrt(sigma2 * h)`) as `NaiveForecaster`, an approximation rather than
    the exact ETS(A,N,N) state-space interval that R's implementation derives
    from the SES equivalence proven by Hyndman & Billah (2003), "Unmasking
    the Theta method", International Journal of Forecasting 19(2):287-290.

    Supported indexes: any index accepted by `future_index` (`PeriodIndex`,
    `DatetimeIndex` with a regular frequency, `RangeIndex`, or numeric
    `Index`). Minimum sample size: 4 observations, or `2 * seasonal_period`
    when seasonal.

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import ThetaForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = ThetaForecaster().fit(y)
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.05, 14.49]

    Pass ``seasonal_period`` to deseasonalize first (multiplicative
    decomposition) and reseasonalize the forecast afterward; this requires
    strictly positive values and at least two full seasonal cycles, e.g.
    ``ThetaForecaster(seasonal_period=12)`` on two years of monthly data.

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - A strong, fast default before reaching for a full state-space
           model; a good general-purpose replacement for the baselines
       * - Avoid when
         - You need exact parity with R's ``forecast::thetaf`` intervals,
           or a model that supports exogenous regressors
       * - Handles trend
         - Yes (linear long-term trend line)
       * - Handles seasonality
         - Yes, via ``seasonal_period`` (multiplicative decomposition)
       * - Extra dependencies
         - None
       * - Min. observations
         - 4, or ``2 * seasonal_period`` when seasonal
    """

    def __init__(self, seasonal_period: int | None = None):
        self.seasonal_period = seasonal_period

    def _fit(self, y: pd.Series) -> None:
        n = len(y)
        min_n = 2 * self.seasonal_period if self.seasonal_period else 4
        if n < min_n:
            raise ValueError(f"ThetaForecaster requires at least {min_n} observations")
        values = y.to_numpy(dtype=float)

        if self.seasonal_period:
            if np.any(values <= 0):
                raise ValueError("seasonal ThetaForecaster requires strictly positive values")
            decomposition = seasonal_decompose(
                values,
                model="multiplicative",
                period=self.seasonal_period,
                extrapolate_trend="freq",
            )
            self.seasonal_index_ = np.asarray(decomposition.seasonal[: self.seasonal_period])
            deseasonalized = values / np.asarray(decomposition.seasonal)
        else:
            self.seasonal_index_ = None
            deseasonalized = values

        t = np.arange(1, n + 1, dtype=float)
        slope, intercept = np.polyfit(t, deseasonalized, 1)
        self.trend_intercept_, self.trend_slope_ = float(intercept), float(slope)
        trend_line = intercept + slope * t
        theta2_line = 2 * deseasonalized - trend_line

        ses = SimpleExpSmoothing(theta2_line, initialization_method="estimated").fit(optimized=True)
        self.alpha_ = float(ses.params["smoothing_level"])
        self._ses_result = ses
        self._n = n
        self._deseasonalized_fitted = 0.5 * np.asarray(ses.fittedvalues) + 0.5 * trend_line

    def _fitted_values(self) -> np.ndarray:
        fitted = self._deseasonalized_fitted
        if self.seasonal_index_ is not None:
            reps = int(np.ceil(self._n / self.seasonal_period))
            season = np.tile(self.seasonal_index_, reps)[: self._n]
            fitted = fitted * season
        return fitted

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        t_future = np.arange(self._n + 1, self._n + horizon + 1, dtype=float)
        trend_forecast = self.trend_intercept_ + self.trend_slope_ * t_future
        ses_forecast = np.asarray(self._ses_result.forecast(horizon))
        mean = 0.5 * ses_forecast + 0.5 * trend_forecast
        if self.seasonal_index_ is not None:
            idx = np.arange(self._n, self._n + horizon) % self.seasonal_period
            mean = mean * self.seasonal_index_[idx]
        steps = np.arange(1, horizon + 1)
        se = np.sqrt(self.sigma2_ * steps)
        return mean, se
