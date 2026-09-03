# ETSForecaster

```{eval-rst}
.. autoclass:: omnicast.ETSForecaster
   :no-index:
```

Error-trend-seasonal exponential smoothing as a state-space model
(`statsmodels.tsa.exponential_smoothing.ets.ETSModel`), with additive error
by construction. Give it explicit knowledge of the series' trend and
seasonal structure rather than having it searched for you (see
{doc}`auto_arima` / {doc}`auto_forecaster` for automatic selection).

```python
from omnicast import ETSForecaster

model = ETSForecaster(
    trend="add",
    seasonal="add",
    seasonal_period=12,
    damped_trend=False,
).fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  207.05    205.12    208.99    204.09    210.01
2026-02  215.37    213.43    217.31    212.41    218.33
2026-03  220.41    218.47    222.35    217.44    223.38
2026-04  225.69    223.74    227.63    222.71    228.66
2026-05  226.02    224.07    227.97    223.03    229.01
2026-06  224.21    222.25    226.17    221.21    227.21
```

Because it's a proper state-space fit rather than an approximation, `ETSForecaster`
exposes the full statistical estimator surface:

```python
model.params_                                # fitted smoothing/seasonal parameters
model.parameter_confidence_intervals_        # 95% CIs for each parameter
model.aic_, model.bic_                       # for comparing against ARIMA, Theta, etc.
```

`trend` and `seasonal` accept `"add"`, `"mul"`, or `None`. Passing
`seasonal="add"` (or `"mul"`) without a `seasonal_period` raises
`ValueError` -- the model needs to know the cycle length, it won't guess it.
Set `damped_trend=True` to flatten long-horizon trend extrapolation, useful
when a linear trend is implausible far into the future.
