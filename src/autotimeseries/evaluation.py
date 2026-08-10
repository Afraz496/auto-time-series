"""Rolling-origin evaluation and automatic model selection."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .metrics import METRICS
from .utils import validate_series


def backtest(
    model, y, horizon: int = 1, initial: int | None = None, step: int = 1, metric: str = "rmse"
) -> pd.DataFrame:
    """Evaluate a forecaster on expanding-window temporal splits."""
    series = validate_series(y)
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}")
    initial = initial or max(10, len(series) // 2)
    if initial < 2 or initial + horizon > len(series):
        raise ValueError("initial and horizon leave no validation observations")
    rows = []
    for end in range(initial, len(series) - horizon + 1, step):
        fitted = deepcopy(model).fit(series.iloc[:end])
        predicted = fitted.predict(horizon, level=95).mean
        actual = series.iloc[end : end + horizon]
        rows.append(
            {
                "cutoff": series.index[end - 1],
                "score": METRICS[metric](actual.to_numpy(), predicted.to_numpy()),
                "n_train": end,
            }
        )
    return pd.DataFrame(rows)
