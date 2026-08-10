# auto-time-series

Automatic statistical forecasting for Python with one consistent, interval-aware API.

```{admonition} Status
:class: note
v0.1 alpha. The API is usable, but model coverage and R parity fixtures are still growing.
```

Every estimator in this library follows the same shape:

```python
model = SomeForecaster(...)
model.fit(y)
forecast = model.predict(horizon=6, level=[80, 95])
```

`y` is a `pandas.Series` (or anything coercible to one) indexed by a `PeriodIndex`,
a `DatetimeIndex` with a regular frequency, a `RangeIndex`, or a plain numeric index.
`forecast` is a {class}`~autotimeseries.ForecastResult` carrying the point forecast
and every requested prediction interval.

## Contents

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
evaluation
```

```{toctree}
:maxdepth: 2
:caption: Model guide & examples

examples/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

api/index
```
