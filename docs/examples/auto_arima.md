# AutoARIMAForecaster

```{eval-rst}
.. autoclass:: omnicast.AutoARIMAForecaster
   :no-index:
```

Grid-searches `(p, d, q)` and, if `seasonal_period` is given, `(P, D, Q, m)`
too, fitting a `SARIMAX` for every combination and keeping the one with the
best information criterion (`"aicc"` by default -- corrected AIC, more
reliable than raw AIC on short series). Failed fits (non-convergent,
singular) are silently skipped rather than aborting the search.

```python
from omnicast import AutoARIMAForecaster

model = AutoARIMAForecaster(
    seasonal_period=12,
    max_p=2, max_d=1, max_q=2,
    max_P=1, max_D=1, max_Q=1,
    information_criterion="aicc",
).fit(y)

print("chosen order:", model.order_, "seasonal_order:", model.seasonal_order_)
print(model.search_results_.head(5))
```

```text
chosen order: (0, 1, 2) seasonal_order: (0, 1, 1, 12)
       order seasonal_order    aic   aicc    bic
0  (0, 1, 2)  (0, 1, 1, 12)  73.75  74.68  77.73
1  (2, 0, 2)  (1, 1, 1, 12)  90.70  93.50  98.01
2  (0, 1, 2)  (1, 1, 1, 12)  92.28  93.71  97.26
3  (1, 0, 2)  (0, 1, 1, 12)  92.82  94.25  98.04
4  (0, 1, 1)  (0, 1, 1, 12)  94.10  94.64  97.23
```

`search_results_` is sorted by `information_criterion` -- the chosen
`(0, 1, 2)(0, 1, 1, 12)` beats the runner-up by a wide AICc margin here,
because it correctly captures both the trend (`d=1`) and yearly seasonal
differencing (`D=1`).

```python
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
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

`AutoARIMAForecaster` subclasses {class}`~omnicast.ARIMAForecaster`, so
after `fit` it exposes the same `params_`, `aic_`, `bic_`, and prediction
API -- plus `order_`/`seasonal_order_` (the winning search result) and
`search_results_` (every candidate tried).

```{admonition} Cost
:class: note
The search space is `(max_p+1) * (max_d+1) * (max_q+1) * (max_P+1) * (max_D+1) * (max_Q+1)`
SARIMAX fits. Keep the `max_*` bounds small for interactive use, or expect
this to take noticeably longer than any other model in the package -- it's
also why it is the default first candidate `AutoForecaster` tries, but the
one most likely to dominate its total runtime.
```
