# AutoForecaster

```{eval-rst}
.. autoclass:: autotimeseries.AutoForecaster
   :no-index:
```

Fits a panel of candidate models, scores each with rolling-origin
backtesting ({func}`~autotimeseries.backtest`), and refits the winner on the
full series. This is the model to reach for by default -- everything else in
this guide is either a candidate it already tries or a tool for building
your own candidate list.

```python
from autotimeseries import AutoForecaster

model = AutoForecaster(
    seasonal_period=12,
    metric="rmse",
    validation_horizon=1,
).fit(y)

print(model.leaderboard_)
```

```text
                     model   score status
0      AutoARIMAForecaster   1.141     ok
1            ETSForecaster   4.341     ok
2          ThetaForecaster   4.665     ok
3          DriftForecaster   4.687     ok
4          NaiveForecaster   4.850     ok
5  SeasonalNaiveForecaster  27.483     ok
6           MeanForecaster  46.988     ok
```

The ranking here lines up with what the individual model pages show:
`AutoARIMAForecaster` and `ETSForecaster` both model the trend and yearly
seasonality directly and backtest far better than the flat `MeanForecaster`
or `SeasonalNaiveForecaster` baselines. `predict` delegates to whichever
model won:

```python
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())     # identical to AutoARIMAForecaster's own output
print(model.best_model_)       # the winning fitted estimator
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  207.64    206.13    209.15    205.33    209.95
2026-02  215.19    213.46    216.92    212.54    217.84
2026-03  219.77    218.00    221.55    217.05    222.50
2026-04  224.87    223.04    226.71    222.07    227.68
2026-05  225.02    223.14    226.91    222.15    227.90
2026-06  223.05    221.12    224.98    220.10    226.00
```

## Default candidates

With no `models=` argument, `AutoForecaster` builds:
`NaiveForecaster`, `MeanForecaster`, `DriftForecaster`, `ThetaForecaster`,
`ETSForecaster`, `AutoARIMAForecaster(seasonal_period=...)` -- plus
`SeasonalNaiveForecaster` inserted whenever `seasonal_period` is set and the
series has more than one full cycle. `LSTMForecaster` is never a default
candidate (see {doc}`lstm`); include it explicitly:

```python
from autotimeseries import LSTMForecaster, ThetaForecaster, AutoForecaster

model = AutoForecaster(
    models=[LSTMForecaster(seed=0), ThetaForecaster(seasonal_period=12)],
).fit(y)
```

Passing `models=[...]` replaces the default panel entirely rather than
extending it -- list every candidate you want considered.

## Resilience to candidate failure

A candidate that raises during backtesting (numerical failure, non-
convergence, insufficient data for its minimum sample size) does not abort
selection: it's recorded in `leaderboard_` with `score = inf` and the
exception message in `status`, and selection proceeds among the rest.
`AutoForecaster.fit` only raises if *every* candidate fails.

`validation_horizon` controls the backtest's forecast horizon per fold (not
the horizon you'll eventually call `predict` with -- those are independent).
`initial` for the internal backtest is chosen automatically as
`max(5, len(y) - max(3 * validation_horizon, len(y) // 4))`.
