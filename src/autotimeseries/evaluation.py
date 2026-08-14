"""Rolling-origin evaluation and temporal backtesting."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import pandas as pd

from .base import BaseForecaster
from .metrics import METRICS
from .result import BacktestResult
from .utils import validate_series

__all__ = ["backtest"]


def backtest(
    model: BaseForecaster,
    y: Iterable[float] | pd.Series,
    horizon: int = 1,
    initial: int | None = None,
    step: int = 1,
    metric: str = "rmse",
) -> BacktestResult:
    """Evaluate a forecaster on expanding-window temporal splits."""
    series = validate_series(y)
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}")
    initial = initial or max(10, len(series) // 2)
    if initial < 2 or initial + horizon > len(series):
        raise ValueError("initial and horizon leave no validation observations")

    rows = []
    folds = []
    model_name = type(model).__name__

    for end in range(initial, len(series) - horizon + 1, step):
        cutoff = series.index[end - 1]
        fitted = deepcopy(model).fit(series.iloc[:end])
        fc = fitted.predict(horizon, level=80)
        actual = series.iloc[end : end + horizon]
        rows.append(
            {
                "cutoff": cutoff,
                "score": float(METRICS[metric](actual.to_numpy(), fc.mean.to_numpy())),
                "n_train": end,
            }
        )
        fold = fc.to_frame().rename(columns={"mean": "predicted"})
        fold.insert(0, "cutoff", cutoff)
        fold.insert(1, "target_date", fold.index)
        fold.insert(2, "step", range(1, len(fold) + 1))
        fold["actual"] = actual.to_numpy()
        folds.append(fold.reset_index(drop=True))

    return BacktestResult(
        scores=pd.DataFrame(rows),
        predictions=pd.concat(folds, ignore_index=True),
        model_name=model_name,
        metric=metric,
    )
