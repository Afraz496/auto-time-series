# Quickstart

```python
import pandas as pd
from omnicast import AutoForecaster

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

## The pieces

**Input.** `y` is a `pandas.Series` with a `PeriodIndex`, a regular-frequency
`DatetimeIndex`, a `RangeIndex`, or a plain numeric index. Models never impute
missing values or guess an irregular frequency for you -- `fit` raises instead.

**Fitting.** `model.fit(y)` learns parameters and populates trailing-underscore
attributes:

- `fitted_values_`, `residuals_`, `sigma2_` -- always present.
- `params_`, `parameter_confidence_intervals_`, `aic_`, `bic_` -- statistical
  estimators only (`ETSForecaster`, `ARIMAForecaster`, `AutoARIMAForecaster`).

**Prediction.** `model.predict(horizon, level=[80, 95])` returns a
{class}`~omnicast.ForecastResult`: a `mean` series plus a lower/upper
series per requested coverage level. `level` accepts a single number or an
iterable; intervals are computed fresh on every call because they depend on
horizon and coverage.

```python
forecast.mean            # point forecast, pandas.Series
forecast.interval(95)    # DataFrame with lower/upper columns for one level
forecast.to_frame()      # mean + every requested interval, one DataFrame
```

**Model selection.** {class}`~omnicast.AutoForecaster` fits a panel of
candidates, scores each with rolling-origin backtesting
({func}`~omnicast.backtest`), and refits the winner on the full series.
Inspect `model.leaderboard_` to see every candidate's score (candidates that
raised an exception show `score = inf` and the error message, rather than
aborting selection).

## Choosing a model

| If you need... | Reach for... |
|---|---|
| A zero-assumption baseline | {doc}`examples/naive` |
| A baseline for seasonal data | {doc}`examples/seasonal_naive` |
| A stable low-variance baseline | {doc}`examples/mean` |
| A trending baseline, cheap to compute | {doc}`examples/drift` |
| A strong, fast, seasonality-aware default | {doc}`examples/theta` |
| Explicit control of error/trend/seasonal structure | {doc}`examples/ets` |
| ARIMA/SARIMA with a known or fixed order, or exogenous regressors | {doc}`examples/arima` |
| ARIMA with the order selected for you | {doc}`examples/auto_arima` |
| A nonlinear/neural model for longer series | {doc}`examples/lstm` |
| Not having to choose at all | {doc}`examples/auto_forecaster` |

See {doc}`evaluation` for backtesting and accuracy metrics.
