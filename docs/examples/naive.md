# NaiveForecaster

```{eval-rst}
.. autoclass:: autotimeseries.NaiveForecaster
   :no-index:
```

The simplest possible forecast: repeat the last observed value, with
horizon-scaled Gaussian intervals (`sqrt(sigma2 * h)`, i.e. random-walk
variance growth). Use it as the floor every other model must beat -- if a
fancier model can't out-backtest `NaiveForecaster`, it isn't earning its
complexity.

```python
from autotimeseries import NaiveForecaster

# y is the sample series defined on the model guide's index page
model = NaiveForecaster().fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
          mean  lower_80  upper_80  lower_95  upper_95
2026-01  197.6    190.74    204.46    187.11    208.09
2026-02  197.6    187.90    207.30    182.76    212.44
2026-03  197.6    185.72    209.48    179.43    215.77
2026-04  197.6    183.88    211.32    176.61    218.59
2026-05  197.6    182.26    212.94    174.14    221.06
2026-06  197.6    180.79    214.41    171.90    223.30
```

The point forecast is flat at the last observed value (`197.6`), and the
interval widens with the square root of the horizon -- exactly as random-walk
theory predicts. `fitted_values_[0]` is `NaN` because there is no prior
observation to predict the first point from:

```python
model.fitted_values_.head(2)
# 2022-01      NaN
# 2022-02    100.0
```
