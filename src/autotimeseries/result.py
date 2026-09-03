"""Forecast and backtest result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["BacktestResult", "ForecastResult"]


@dataclass(frozen=True)
class ForecastResult:
    """Point forecasts and prediction intervals in tidy, labelled form."""

    mean: pd.Series
    lower: dict[float, pd.Series]
    upper: dict[float, pd.Series]
    model_name: str
    observed: pd.Series | None = None

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

    def plot(self, observed: pd.Series | None = None, title: str | None = None, **kwargs: Any) -> Axes:
        """Plot the forecast trajectory with its prediction bands over the observed history."""
        from .plotting import plot_forecast_trajectories

        return plot_forecast_trajectories(
            self,
            observed=self.observed if observed is None else observed,
            title=title or f"{self.model_name} Forecast",
            **kwargs,
        )


@dataclass(frozen=True)
class BacktestResult:
    """Rolling-origin backtest scores plus the predictions behind them.

    Parameters
    ----------
    scores : pd.DataFrame
        One row per fold: 'cutoff', 'score', 'n_train'.
    predictions : pd.DataFrame
        One row per fold-step: 'cutoff', 'target_date', 'step', 'predicted',
        'actual', and interval columns 'lower_{level}' / 'upper_{level}'.
    model_name : str
        Name of the evaluated model.
    metric : str
        Metric used for scoring (e.g. 'rmse').
    observed : pd.Series, optional
        The series that was backtested; ``.plot()`` falls back to it.
    """

    scores: pd.DataFrame
    predictions: pd.DataFrame
    model_name: str
    metric: str
    observed: pd.Series | None = None

    def __getitem__(self, key: Any) -> Any:
        return self.scores[key]

    def __len__(self) -> int:
        return len(self.scores)

    def summary(self) -> pd.Series:
        """Descriptive statistics of the fold scores (``pd.Series.describe``)."""
        return self.scores["score"].describe()

    def plot(self, observed: pd.Series | None = None, title: str | None = None, **kwargs: Any) -> Axes:
        """Plot fold predictions against observations. See ``plot_backtest``."""
        from .plotting import plot_backtest

        return plot_backtest(
            self,
            observed=self.observed if observed is None else observed,
            title=title or f"Backtest Predictions ({self.model_name})",
            **kwargs,
        )

