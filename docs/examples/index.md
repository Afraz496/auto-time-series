# Model guide & examples

Every example on the following pages reuses the same sample series: four
years of monthly data with a linear trend and yearly seasonality, generated
once and shared across pages so the outputs are directly comparable.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
t = np.arange(48)
trend = 100 + 2.2 * t
season = 12 * np.sin(2 * np.pi * t / 12)
noise = rng.normal(0, 2, size=48)

y = pd.Series(
    (trend + season + noise).round(1),
    index=pd.period_range("2022-01", periods=48, freq="M"),
    name="sales",
)
```

```text
2022-01    100.0
2022-02    108.8
2022-03    114.2
...
2025-10    183.9
2025-11    192.5
2025-12    197.6
Freq: M, Name: sales, Length: 48, dtype: float64
```

```{toctree}
:maxdepth: 1

naive
seasonal_naive
mean
drift
theta
ets
arima
auto_arima
lstm
auto_forecaster
```
