"""Automatic, interval-aware time-series forecasting."""

from .auto import AutoForecaster
from .base import BaseForecaster
from .evaluation import backtest
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
from .result import ForecastResult

__version__ = "0.1.0"
__all__ = [
    "ARIMAForecaster",
    "AutoARIMAForecaster",
    "AutoForecaster",
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
    "rmse",
    "smape",
]
