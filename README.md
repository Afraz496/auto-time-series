# Auto Time Series

Automatic statistical forecasting for Python with one consistent, interval-aware API.

> **Status:** v0.1 alpha. The API is usable, but model coverage and R parity fixtures are still growing.

## Install

```bash
pip install auto-time-series
```

For local development:

```bash
uv sync --extra dev
```

Full docs with a worked example for every model live under [`docs/`](docs/index.md);
build them locally with:

```bash
uv sync --extra docs
uv run sphinx-build -b html docs docs/_build/html
```

`LSTMForecaster` requires PyTorch, kept out of the base install:

```bash
pip install auto-time-series[torch]
# or, for development:
uv sync --extra dev --extra torch
```

## Quick start

```python
import pandas as pd
from autotimeseries import AutoForecaster

y = pd.Series(
    [112, 118, 121, 130, 128, 137, 143, 149, 154, 162, 169, 175],
    index=pd.period_range("2025-01", periods=12, freq="M"),
)

model = AutoForecaster(
    seasonal_period=None,
    metric="rmse",
    validation_horizon=1,
).fit(y)

forecast = model.predict(horizon=6, level=[80, 95])
print(model.leaderboard_)
print(forecast.to_frame())
```

Every fitted estimator exposes `fitted_values_`, `residuals_`, `sigma2_`, and `prediction_intervals_`. Statistical estimators also expose `params_`, `parameter_confidence_intervals_` (95%), `aic_`, and `bic_`. Prediction intervals are returned on each prediction because they depend on horizon and requested coverage.

## Models

| Estimator | Purpose | Intervals |
|---|---|---|
| `NaiveForecaster` | Random walk | Horizon-scaled Gaussian innovation |
| `SeasonalNaiveForecaster` | Seasonal random walk | Cycle-scaled Gaussian innovation |
| `MeanForecaster` | Historical mean | Mean forecast uncertainty |
| `DriftForecaster` | Random walk with drift | Drift forecast uncertainty |
| `ThetaForecaster` | Theta method (port of R `forecast::thetaf`) | Random-walk innovation scaling |
| `ETSForecaster` | Error/trend/seasonal state space | State-space forecast uncertainty |
| `ARIMAForecaster` | ARIMA/SARIMA, optional regressors | State-space forecast uncertainty |
| `AutoARIMAForecaster` | AICc grid-selected ARIMA | State-space forecast uncertainty |
| `LSTMForecaster` | Autoregressive LSTM (`torch`, optional) | Random-walk innovation scaling |
| `AutoForecaster` | Rolling-origin model selection | Selected model's intervals |

## Evaluation

```python
from autotimeseries import NaiveForecaster, backtest

folds = backtest(NaiveForecaster(), y, horizon=3, initial=6, metric="rmse")
print(folds)
```

Available metrics are MAE, RMSE, MAPE, and sMAPE. Backtesting uses expanding windows and never trains on future observations.

## Design and scope

The package follows pandas index semantics and the familiar `fit`/`predict` estimator pattern. Learned state uses trailing underscores. Models validate input rather than silently imputing data or guessing an irregular date frequency.

This codebase is a Python implementation foundation, not a blanket claim of parity with R forecasting packages. Each future port must record its algorithm source, licensing, deviations, and numerical parity tests. See [CONTRIBUTING.md](CONTRIBUTING.md).

`ThetaForecaster` is the first R port: a compatible pure-Python reimplementation of `forecast::thetaf`'s classical Theta method, described in its own docstring along with the exact deviations from R's output (approximate intervals, no numerical parity fixtures yet).

`LSTMForecaster` is the first wrapper around a Python deep-learning module (`torch`, optional dependency), following the same `BaseForecaster` interface as the statsmodels-backed models. It is not part of `AutoForecaster`'s default candidate list -- pass it explicitly via `AutoForecaster(models=[...])` -- since it is optional-dependency and materially slower to backtest.

Licensed under Apache-2.0.
