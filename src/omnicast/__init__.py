"""Automatic, interval-aware time-series forecasting."""

from .auto import AutoForecaster
from .base import BaseForecaster
from .evaluation import Backtester, backtest
from .metrics import mae, mape, rmse, smape
from .models import (
    ARIMAForecaster,
    AutoARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    LSTMForecaster,
    MeanForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)
from .plotting import plot_backtest, plot_metric_by_horizon
from .result import BacktestResult, ForecastResult

__version__ = "0.1.1"
__all__ = [
    "ARIMAForecaster",
    "AutoARIMAForecaster",
    "AutoForecaster",
    "BacktestResult",
    "Backtester",
    "BaseForecaster",
    "DriftForecaster",
    "ETSForecaster",
    "ForecastResult",
    "LSTMForecaster",
    "MeanForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ThetaForecaster",
    "backtest",
    "mae",
    "mape",
    "plot_backtest",
    "plot_metric_by_horizon",
    "rmse",
    "smape",
]
