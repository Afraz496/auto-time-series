# LSTMForecaster

```{eval-rst}
.. autoclass:: autotimeseries.LSTMForecaster
   :no-index:
```

The package's first neural model: wraps `torch.nn.LSTM`, following the same
`BaseForecaster` interface as the statsmodels-backed estimators. Requires the
optional `torch` extra:

```bash
pip install auto-time-series[torch]
```

If `torch` isn't installed, the module still imports cleanly, and only
`fit()` raises a clear `ImportError` naming the missing package.

```python
from autotimeseries import LSTMForecaster

model = LSTMForecaster(
    lookback=12,      # window of past points used to predict the next one
    hidden_size=16,
    num_layers=1,
    epochs=150,
    learning_rate=1e-2,
    seed=0,
).fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
print(forecast.to_frame())
```

```text
           mean  lower_80  upper_80  lower_95  upper_95
2026-01  204.24    201.46    207.03    199.98    208.50
2026-02  209.65    205.70    213.59    203.62    215.67
2026-03  212.22    207.40    217.05    204.84    219.61
2026-04  212.16    206.58    217.73    203.63    220.68
2026-05  210.43    204.20    216.66    200.90    219.96
2026-06  207.81    200.99    214.64    197.37    218.25
```

The series is standardized (zero mean, unit variance, from training data
only) and reframed as sliding windows of length `lookback` predicting the
next point; training is full-batch Adam/MSE for `epochs` iterations.
Multi-step forecasts are autoregressive -- each predicted point is fed back
in as the newest observation of the next window, so errors can compound over
a long horizon the way they do for any autoregressive model.

```{admonition} Not in AutoForecaster's default candidates
:class: note
`LSTMForecaster` is optional-dependency and materially slower to backtest
than the built-in statistical models, so `AutoForecaster` never selects it
automatically. Include it explicitly:

    AutoForecaster(models=[LSTMForecaster(), ThetaForecaster(seasonal_period=12)])
```

```{admonition} Reproducibility
:class: warning
`seed` fixes `torch.manual_seed`, but training is not guaranteed deterministic
across torch versions or hardware (CPU/GPU, BLAS backend). Don't rely on
bit-exact reproduction across environments.
```

Minimum sample size is `lookback + 2` observations; `fit` raises
`ValueError` below that.
