# Evaluation

## Backtesting

{func}`~autotimeseries.backtest` scores a forecaster on expanding-window,
rolling-origin splits: it fits on `y[:end]`, forecasts `horizon` steps, scores
against the true values, then slides `end` forward by `step` and repeats.
It never trains on future observations.

```python
from autotimeseries import NaiveForecaster, backtest

folds = backtest(
    NaiveForecaster(),
    y,
    horizon=3,      # steps forecast per fold
    initial=24,     # size of the first training window
    step=1,         # how far the origin advances between folds
    metric="rmse",  # "mae" | "rmse" | "mape" | "smape"
)
print(folds)
#       cutoff  score  n_train
# 0  2023-12    3.41       24
# 1  2024-01    3.58       25
# ...
```

Each row is one fold: the cutoff timestamp, the fold's score, and the
training-window size at that point. Average `folds["score"]` for a single
summary number.

`initial` defaults to `max(10, len(y) // 2)` when omitted. `backtest` raises
`ValueError` if `initial` and `horizon` leave no room for a single validation
fold, and `ValueError` for an unknown `metric` name.

This is exactly the mechanism {class}`~autotimeseries.AutoForecaster` uses
internally to rank candidates -- see {doc}`examples/auto_forecaster`.

## Metrics

Full signatures are in the {doc}`api/metrics` reference. Each metric takes
`actual` and `predicted` array-likes of equal length and
returns a single float:

```python
from autotimeseries import mae, rmse, mape, smape

actual = [100, 110, 90]
predicted = [98, 115, 95]

mae(actual, predicted)     # 5.0
rmse(actual, predicted)    # 5.35...
mape(actual, predicted)    # percentage error; raises ValueError if actual has a zero
smape(actual, predicted)   # symmetric percentage error, 0 when both are 0
```

`mape` raises `ValueError` when any `actual` value is zero (undefined
denominator). Use `smape` for series that legitimately cross zero.
