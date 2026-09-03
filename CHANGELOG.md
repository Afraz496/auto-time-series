# Changelog

## 0.1.2 - 2026-09-03

### Added

- Docs are now hosted at [afraz496.github.io/omnicast](https://afraz496.github.io/omnicast/), deployed by a new `.github/workflows/pages.yml` on every push to `main`. Added as a `Documentation` link in `pyproject.toml`'s `project.urls`, so it shows up on the PyPI project page.

## 0.1.1 - 2026-09-03

### Added

- The real-data notebook, [`examples/epidatpy_forecasting_and_plotting.ipynb`](examples/epidatpy_forecasting_and_plotting.ipynb), is now part of the Sphinx docs site (via `myst-nb`), linked from the [model guide](docs/examples/index.md) under "Real-data walkthrough". Its saved outputs (including plots) are rendered as-is rather than re-executed on every docs build, so the build doesn't depend on network access or the optional `epidatpy` package.

## 0.1.0 - 2026-09-03

**Renamed from `auto-time-series` to `omnicast`.** Same codebase, same repo (GitHub auto-redirects the old URL), new identity: this project's goal is to be a definitive, unified wrapper over classical and deep-learning forecasting methods -- one consistent, interval-aware `fit`/`predict` API, with backtesting and plotting as first-class citizens rather than an afterthought. Since `omnicast` is a new, unpublished PyPI project, versioning restarts at `0.1.0` rather than continuing `auto-time-series`'s `0.2.0`. Everything from that release carries over unchanged; see [Prior history](#prior-history-as-auto-time-series) below for what shipped before the rename.

### Added

- Plotting (`omnicast.plotting`, matplotlib + seaborn, now base dependencies): `ForecastResult.plot()`, `BacktestResult.plot()`, `AutoForecaster.plot_all()`, plus the standalone `plot_backtest()` and `plot_metric_by_horizon()`. `backtest()` is now a thin wrapper over a new public `Backtester` class.
- Sphinx documentation site (`docs/`): installation, quickstart, evaluation guide, a worked example for every model ([`docs/examples/`](docs/examples/index.md)), and full autodoc API reference ([`docs/api/`](docs/api/index.md)). Built in CI.
- A numpydoc `Examples` section (verified as a real doctest, not hand-typed) and a "when to use this model" comparison table (best for / avoid when / trend / seasonality / extra dependencies / min. observations) on every forecaster's class docstring: `NaiveForecaster`, `MeanForecaster`, `DriftForecaster`, `SeasonalNaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `ARIMAForecaster`, `AutoARIMAForecaster`, and `LSTMForecaster` (rendered in [`docs/api/models.md`](docs/api/models.md)), plus `AutoForecaster` (rendered in [`docs/api/auto.md`](docs/api/auto.md)). `pytest --doctest-modules src/omnicast` runs in CI so these can't silently drift from the code.
- An end-to-end real-data walkthrough, [`examples/epidatpy_forecasting_and_plotting.ipynb`](examples/epidatpy_forecasting_and_plotting.ipynb) (see [Datasets used in examples](#datasets-used-in-examples) below).

### Fixed

- `docs` CI job failing to build: `napoleon_numpy_docstring` was `False`, so the plotting module's numpydoc `Parameters` sections were parsed as raw RST instead of being converted by Napoleon, and `bare **kwargs`/undescribed parameters broke under `sphinx-build -W`. `html_static_path` also pointed at a `docs/_static` directory that never existed. Added [`tests/test_docs.py`](tests/test_docs.py), a pytest regression test that runs the Sphinx build itself.
- `docs/api/evaluation.md` documented only the `backtest()` wrapper function; the new, exported `Backtester` class had no entry in the API reference.

### Datasets used in examples

Every per-model page in the [Sphinx model guide](docs/examples/index.md) (`docs/examples/*.md`) and every new docstring `Examples` section use **synthetic, in-repo data** generated with `numpy`/hand-picked literals -- not a real dataset -- chosen for compact, exactly reproducible output. The model guide's shared series is defined once in [`docs/examples/index.md`](docs/examples/index.md); each docstring's series is inline in its own `Examples` section.

The one **real** dataset in the repo is in [`examples/epidatpy_forecasting_and_plotting.ipynb`](examples/epidatpy_forecasting_and_plotting.ipynb):

| Detail | Value |
| --- | --- |
| Data | Weekly % of inpatient insurance claims coded with an influenza diagnosis, California, Jan 2024 - Feb 2026 |
| Source | CMU Delphi Epidata V5 API, `claims_inpatient` source, `claims_inpatient_adm_pct_claims_flu` signal, `geo_type="state"`, `geo_values="ca"` |
| Client | [`epidatpy`](https://github.com/cmu-delphi/epidatpy) (CMU Delphi) |
| Models demonstrated on it | `NaiveForecaster`, `ThetaForecaster`, `ETSForecaster`, `AutoARIMAForecaster`, `AutoForecaster` -- via `.fit`/`.predict`, `backtest()`, and `plot_all()` |

No documentation host (GitHub Pages / Read the Docs) is configured yet, so the links above point at the in-repo doc sources rather than a live site; `uv run sphinx-build -b html docs docs/_build/html` builds them locally.

## Prior history (as `auto-time-series`)

Released under the project's original name, before the rename above. Code references below (`autotimeseries`, `src/autotimeseries`) reflect what the package was actually called at the time.

### 0.2.0

- Plotting (`autotimeseries.plotting`, matplotlib + seaborn, now base dependencies): `ForecastResult.plot()`, `BacktestResult.plot()`, `AutoForecaster.plot_all()`, plus the standalone `plot_backtest()` and `plot_metric_by_horizon()`. `backtest()` became a thin wrapper over a new public `Backtester` class.
- Sphinx documentation site (`docs/`): installation, quickstart, evaluation guide, a worked example for every model, and full autodoc API reference. Built in CI.
- A numpydoc `Examples` section and a "when to use this model" comparison table on every forecaster's class docstring.
- Fixed the `docs` CI job (Napoleon numpy-docstring config, missing `docs/_static`) and documented the `Backtester` class in the API reference.

### 0.1.0

- Common fit/predict API and labelled forecast result objects.
- Naive, seasonal-naive, mean, drift, Theta, ETS, ARIMA, automatic ARIMA, and LSTM models.
- `ThetaForecaster`: the first R port, a compatible reimplementation of `forecast::thetaf`.
- `LSTMForecaster`: the first optional-dependency deep-learning wrapper (`torch`).
- Prediction intervals, residuals, fitted values, model parameters, AIC, and BIC.
- Rolling-origin evaluation, accuracy metrics, and automatic model selection.
- Fixed `AutoForecaster.fit` mutating a caller-supplied `models` list on each call.
- CI workflow running ruff and pytest across Python 3.10-3.12.
