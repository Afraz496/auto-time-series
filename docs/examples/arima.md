# ARIMAForecaster

```{eval-rst}
.. autoclass:: omnicast.ARIMAForecaster
   :no-index:
```

A direct wrapper around `statsmodels.tsa.statespace.sarimax.SARIMAX`. Use it
when you already know (or want to fix) the ARIMA order -- for a searched
order, see {doc}`auto_arima`.

```python
from omnicast import ARIMAForecaster

model = ARIMAForecaster(order=(2, 1, 1)).fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  201.99    196.56    207.42    193.68    210.29
2026-02  204.43    194.68    214.18    189.52    219.34
2026-03  206.33    192.07    220.58    184.53    228.12
2026-04  207.46    189.04    225.88    179.29    235.64
2026-05  208.30    185.94    230.65    174.11    242.49
2026-06  208.82    182.83    234.81    169.07    248.57
```

`order=(2, 1, 1)` alone can't represent the series' yearly seasonality (note
the wide, fast-growing intervals compared to {doc}`ets`); pass
`seasonal_order=(P, D, Q, m)` for SARIMA:

```python
model = ARIMAForecaster(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
).fit(y)
```

## Exogenous regressors

Unlike most models in this package, `ARIMAForecaster` supports exogenous
regressors -- pass a `X` with a matching index at both `fit` and `predict`
time (future values for the forecast horizon must be supplied; the model
will not extrapolate them for you):

```python
model = ARIMAForecaster(order=(1, 0, 0)).fit(y, X=X_train)
forecast = model.predict(horizon=6, X=X_future)
```

Fitted statistical attributes are available exactly as with
{doc}`ets`: `params_`, `parameter_confidence_intervals_`, `aic_`, `bic_`.
