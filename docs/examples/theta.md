# ThetaForecaster

```{eval-rst}
.. autoclass:: autotimeseries.ThetaForecaster
   :no-index:
```

The package's first R port: a compatible pure-Python reimplementation of R's
`forecast::thetaf` (classical Theta method, Assimakopoulos & Nikolopoulos
2000). It decomposes the series into a linear long-term trend and a
curvature-doubled "theta line" extrapolated with simple exponential
smoothing, then averages the two. Strong, fast, and a good default before
reaching for a full state-space model.

## Non-seasonal

```python
from autotimeseries import ThetaForecaster

model = ThetaForecaster().fit(y)          # seasonal_period=None
forecast = model.predict(horizon=6)
```

## Seasonal

Pass `seasonal_period` to deseasonalize first (multiplicative classical
decomposition) and reseasonalize the forecast afterward. This requires
strictly positive values and at least two full seasonal cycles.

```python
model = ThetaForecaster(seasonal_period=12).fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  206.46    203.31    209.60    201.65    211.27
2026-02  216.66    212.21    221.11    209.86    223.46
2026-03  222.14    216.69    227.59    213.81    230.47
2026-04  227.11    220.82    233.40    217.49    236.73
2026-05  225.00    217.97    232.03    214.25    235.76
2026-06  219.82    212.11    227.52    208.03    231.60
```

Unlike the flat `DriftForecaster` trend, Theta's forecast tracks both the
upward trend and the yearly seasonal shape. `alpha_` is the fitted
exponential-smoothing weight for the theta=2 line:

```python
model.alpha_   # smoothing_level chosen by SES's own MLE, not a hyperparameter you set
```

```{admonition} Known deviation from R
:class: warning
Prediction intervals here use the same residual-variance random-walk scaling
(`sqrt(sigma2 * h)`) as `NaiveForecaster`, not the exact ETS(A,N,N)
state-space interval R's `thetaf` derives from the SES equivalence (Hyndman
& Billah 2003). Point forecasts follow the same method; interval widths will
differ slightly. See `CONTRIBUTING.md` for the parity-fixture policy.
```

Minimum sample size is 4 observations (or `2 * seasonal_period` when
seasonal); `fit` raises `ValueError` below that, or if seasonal values are
non-positive.
