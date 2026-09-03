"""Automatic selection across heterogeneous forecasting models."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .base import BaseForecaster
from .evaluation import Backtester
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
    """Choose the model with the best rolling-origin validation score.

    With ``keep_all=True`` every candidate is also fitted on the full series and
    kept in ``fitted_`` (name -> fitted model).

    Examples
    --------
    >>> import pandas as pd
    >>> from omnicast import AutoForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0])
    >>> model = AutoForecaster().fit(y)
    >>> model.leaderboard_["model"].iloc[0]
    'ThetaForecaster'
    >>> model.predict(horizon=2).mean.round(2).tolist()
    [14.05, 14.49]

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - The default entry point -- backtests a panel of candidates and
           refits the winner, so you don't have to pick a model by hand
       * - Avoid when
         - You already know which model fits, need exogenous regressors
           (not yet supported here), or want a single deterministic model
           without a backtest step
       * - Handles trend
         - Depends on which candidate wins the backtest
       * - Handles seasonality
         - Yes, via ``seasonal_period`` (adds seasonal-aware candidates)
       * - Extra dependencies
         - None by default; include :class:`~omnicast.LSTMForecaster`
           explicitly via ``models=[...]`` to pull in ``torch``
       * - Min. observations
         - Whatever the strictest candidate in ``models`` requires; a
           failing candidate is skipped rather than aborting selection
    """

    def __init__(
        self,
        models: list[BaseForecaster] | None = None,
        seasonal_period: int | None = None,
        metric: str = "rmse",
        validation_horizon: int = 1,
        keep_all: bool = False,
    ):
        self.models = models
        self.seasonal_period = seasonal_period
        self.metric = metric
        self.validation_horizon = validation_horizon
        self.keep_all = keep_all

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
        bt = Backtester(horizon=self.validation_horizon, initial=initial, metric=self.metric)
        for model in candidates:
            try:
                score = float(bt.run(model, series)["score"].mean())
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
        self.candidates_ = [model for _, model in successful]
        self.leaderboard_ = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
        best = min(successful, key=lambda item: item[0])[1]
        if self.keep_all:
            self.fitted_ = {
                type(m).__name__: deepcopy(m).fit(series) for m in self.candidates_
            }
            self.best_model_ = self.fitted_[type(best).__name__]
        else:
            self.fitted_ = None
            self.best_model_ = deepcopy(best).fit(series)
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

    def plot_all(
        self, horizon=None, level=(80, 95), observed=None, intervals=False, **kwargs
    ):
        """Overlay every successful candidate's forecast on one axes.

        Refits each candidate on the full training series, unless the estimator
        was built with ``keep_all=True`` (then the stored fits are reused).

        Parameters
        ----------
        horizon : int, optional
            Steps to forecast. Defaults to ``validation_horizon``.
        level : float or sequence of float, default (80, 95)
            Prediction-interval coverage levels to compute.
        observed : pd.Series, optional
            History to draw. Defaults to the full training series.
        intervals : bool, default False
            Shade each candidate's prediction bands. Off by default -- with
            several candidates the overlapping bands get muddy.
        **kwargs
            Forwarded to the trajectory plot (`title`, `ax`, `save_path`, ...).
        """
        from .plotting import plot_forecast_trajectories

        self._check_fitted()
        horizon = horizon or self.validation_horizon
        if self.fitted_ is not None:
            forecasts = {n: m.predict(horizon, level=level) for n, m in self.fitted_.items()}
        else:
            forecasts = {
                type(m).__name__: deepcopy(m).fit(self.y_).predict(horizon, level=level)
                for m in self.candidates_
            }
        return plot_forecast_trajectories(
            forecasts,
            observed=self.y_ if observed is None else observed,
            intervals=intervals,
            **kwargs,
        )

    def _fit(self, y):
        pass

    def _fitted_values(self):
        return self.best_model_.fitted_values_.to_numpy()

    def _forecast(self, horizon):
        return self.best_model_._forecast(horizon)
