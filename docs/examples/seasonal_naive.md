# SeasonalNaiveForecaster

```{eval-rst}
.. autoclass:: omnicast.SeasonalNaiveForecaster
   :no-index:
```

Repeats the value from the same point in the last full seasonal cycle instead
of just the last observation. It's the baseline to beat whenever a series has
real seasonal structure -- `NaiveForecaster` will systematically miss the
seasonal swing that this model captures for free.

```python
from omnicast import SeasonalNaiveForecaster

model = SeasonalNaiveForecaster(seasonal_period=12).fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
          mean  lower_80  upper_80  lower_95  upper_95
2026-01  179.1    145.05    213.15    127.02    231.18
2026-02  189.2    155.15    223.25    137.12    241.28
2026-03  192.8    158.75    226.85    140.72    244.88
2026-04  197.6    163.55    231.65    145.52    249.68
2026-05  198.6    164.55    232.65    146.52    250.68
2026-06  196.3    162.25    230.35    144.22    248.38
```

Each forecast reuses the corresponding month from `2025`. Intervals scale
with `sqrt(sigma2 * cycles_ahead)`, so the January-2026 forecast (one cycle
ahead of the December-2025 cutoff for that month) has a narrower interval
than a forecast two full cycles out would.

`seasonal_period` is required and validated eagerly: constructing
`SeasonalNaiveForecaster(seasonal_period=0)` raises `ValueError` before `fit`
is ever called. `fit` additionally requires `len(y) > seasonal_period` -- at
least one full cycle of history.
