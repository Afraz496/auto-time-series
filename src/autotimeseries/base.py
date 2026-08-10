"""Shared estimator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from .result import ForecastResult
from .utils import future_index, validate_levels, validate_series


class BaseForecaster(ABC):
    """Base class for all forecasters.

    Learned attributes follow the scikit-learn trailing-underscore convention.
    """

    def fit(self, y: Iterable[float] | pd.Series, X: object = None) -> BaseForecaster:
        if X is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support exogenous regressors"
            )
        self.y_ = validate_series(y)
        self._fit(self.y_)
        fitted = np.asarray(self._fitted_values(), dtype=float)
        self.fitted_values_ = pd.Series(fitted, index=self.y_.index, name="fitted")
        self.residuals_ = (self.y_ - self.fitted_values_).rename("residual")
        finite = self.residuals_.dropna()
        self.sigma2_ = float(np.mean(np.square(finite))) if len(finite) else 0.0
        self.n_obs_ = len(self.y_)
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: object = None,
        level: float | Iterable[float] = (80, 95),
    ) -> ForecastResult:
        self._check_fitted()
        if X is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support exogenous regressors"
            )
        levels = validate_levels(level)
        index = future_index(self.y_.index, horizon)
        mean, standard_error = self._forecast(horizon)
        mean_series = pd.Series(np.asarray(mean), index=index, name="mean")
        se = np.asarray(standard_error, dtype=float)
        lower, upper = {}, {}
        for coverage in levels:
            z = norm.ppf(0.5 + coverage / 200)
            lower[coverage] = pd.Series(mean_series.to_numpy() - z * se, index=index)
            upper[coverage] = pd.Series(mean_series.to_numpy() + z * se, index=index)
        result = ForecastResult(mean_series, lower, upper, type(self).__name__)
        self.forecast_ = result
        self.prediction_intervals_ = {coverage: result.interval(coverage) for coverage in levels}
        return result

    def fit_predict(
        self, y: Iterable[float] | pd.Series, horizon: int, **kwargs: object
    ) -> ForecastResult:
        return self.fit(y).predict(horizon, **kwargs)

    def get_params(self) -> dict[str, object]:
        return {key: value for key, value in vars(self).items() if not key.endswith("_")}

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit before predict")

    @abstractmethod
    def _fit(self, y: pd.Series) -> None: ...

    @abstractmethod
    def _fitted_values(self) -> np.ndarray: ...

    @abstractmethod
    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]: ...
