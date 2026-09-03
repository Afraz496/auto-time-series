# MeanForecaster

```{eval-rst}
.. autoclass:: omnicast.MeanForecaster
   :no-index:
```

Forecasts the historical mean for every horizon step -- flat, low-variance,
and blind to trend or seasonality. Useful as a stability baseline (does the
series even have a signal worth modeling?) and, via
{class}`~omnicast.AutoForecaster`'s leaderboard, as a sanity check that
should almost always lose to anything trend-aware.

```python
from omnicast import MeanForecaster

model = MeanForecaster().fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  151.02    111.98    190.07     91.32    210.73
2026-02  151.02    111.98    190.07     91.32    210.73
2026-03  151.02    111.98    190.07     91.32    210.73
2026-04  151.02    111.98    190.07     91.32    210.73
2026-05  151.02    111.98    190.07     91.32    210.73
2026-06  151.02    111.98    190.07     91.32    210.73
```

On this trending series `151.02` (the 4-year average) is a poor forecast for
`2026`, which is exactly why the {doc}`auto_forecaster` example ranks
`MeanForecaster` last: the leaderboard scores are backtest RMSE, and a flat
mean can't track trend. The interval is constant across the horizon
(`sqrt(sigma2 * (1 + 1/n))`) because every future point carries the same
uncertainty about the estimated mean -- there's no compounding random-walk
term to grow it with `h`.
