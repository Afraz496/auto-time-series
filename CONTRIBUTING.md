# Contributing

Install the development environment with `uv sync --extra dev`, then run `uv run pytest` and `uv run ruff check .`.

New models subclass `BaseForecaster` and implement `_fit`, `_fitted_values`, and `_forecast`. A model must document its source algorithm, license compatibility, interval assumptions, supported indexes, and minimum sample size. Faithful ports should include parity fixtures generated from the cited R package; compatible reimplementations should say so explicitly.

