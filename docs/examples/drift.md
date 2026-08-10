# DriftForecaster

```{eval-rst}
.. autoclass:: autotimeseries.DriftForecaster
   :no-index:
```

A random walk with drift: extrapolates the straight line between the first
and last observation. Cheap, closed-form, and a much stronger baseline than
`NaiveForecaster` on a trending series with no seasonality.

```python
from autotimeseries import DriftForecaster

model = DriftForecaster().fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  199.68    193.29    206.07    189.90    209.45
2026-02  201.75    192.62    210.89    187.79    215.72
2026-03  203.83    192.53    215.13    186.55    221.11
2026-04  205.91    192.73    219.08    185.76    226.06
2026-05  207.98    193.11    222.86    185.23    230.73
2026-06  210.06    193.61    226.51    184.90    235.22
```

`drift_` is `(y[-1] - y[0]) / (n - 1)` -- the average per-step change over the
whole series -- added to the last observation at each step:

```python
model.drift_   # 2.0766... (~2.08 units/month, close to the synthetic trend of 2.2)
```

Because this series also has yearly seasonality that a straight line can't
capture, `DriftForecaster` will systematically over- or under-shoot depending
on where in the cycle the forecast horizon falls; compare against
{doc}`theta` or {doc}`ets` when seasonality matters. `fit` raises
`ValueError` on a series shorter than two observations, since drift is
undefined otherwise.
