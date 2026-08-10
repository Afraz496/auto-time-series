"""Automatic selection across heterogeneous forecasting models."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .base import BaseForecaster
from .evaluation import backtest
from .models import (
    AutoARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)
from .utils import validate_series


class AutoForecaster(BaseForecaster):
    """Choose the model with the best rolling-origin validation score."""

    def __init__(
        self,
        models: list[BaseForecaster] | None = None,
        seasonal_period: int | None = None,
        metric: str = "rmse",
        validation_horizon: int = 1,
    ):
        self.models = models
        self.seasonal_period = seasonal_period
        self.metric = metric
        self.validation_horizon = validation_horizon

    def fit(self, y, X=None):
        if X is not None:
            raise NotImplementedError(
                "AutoForecaster exogenous model selection is not yet supported"
            )
        series = validate_series(y)
        candidates = list(self.models) if self.models else [
            NaiveForecaster(),
            MeanForecaster(),
            DriftForecaster(),
            ThetaForecaster(),
            ETSForecaster(),
            AutoARIMAForecaster(seasonal_period=self.seasonal_period),
        ]
        if self.seasonal_period and len(series) > self.seasonal_period:
            candidates.insert(1, SeasonalNaiveForecaster(self.seasonal_period))
        rows, successful = [], []
        initial = max(5, len(series) - max(3 * self.validation_horizon, len(series) // 4))
        for model in candidates:
            try:
                scores = backtest(
                    model,
                    series,
                    horizon=self.validation_horizon,
                    initial=initial,
                    metric=self.metric,
                )
                score = float(scores["score"].mean())
                rows.append({"model": type(model).__name__, "score": score, "status": "ok"})
                successful.append((score, model))
            # Candidate libraries expose heterogeneous numerical failure types;
            # one failed candidate must not prevent selection among the rest.
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {"model": type(model).__name__, "score": float("inf"), "status": str(exc)}
                )
        if not successful:
            raise RuntimeError("All candidate models failed")
        self.leaderboard_ = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
        self.best_model_ = deepcopy(min(successful, key=lambda item: item[0])[1]).fit(series)
        self.y_, self.is_fitted_, self.n_obs_ = series, True, len(series)
        self.fitted_values_ = self.best_model_.fitted_values_
        self.residuals_, self.sigma2_ = self.best_model_.residuals_, self.best_model_.sigma2_
        if hasattr(self.best_model_, "parameter_confidence_intervals_"):
            self.parameter_confidence_intervals_ = self.best_model_.parameter_confidence_intervals_
        return self

    def predict(self, horizon, X=None, level=(80, 95)):
        self._check_fitted()
        result = self.best_model_.predict(horizon, X=X, level=level)
        self.forecast_, self.prediction_intervals_ = result, self.best_model_.prediction_intervals_
        return result

    def _fit(self, y):
        pass

    def _fitted_values(self):
        return self.best_model_.fitted_values_.to_numpy()

    def _forecast(self, horizon):
        return self.best_model_._forecast(horizon)
