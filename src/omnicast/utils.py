"""Input and index helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def validate_series(y: Iterable[float] | pd.Series) -> pd.Series:
    """Convert input to a finite, non-empty float Series."""
    series = y.copy() if isinstance(y, pd.Series) else pd.Series(y)
    if series.empty:
        raise ValueError("y must contain at least one observation")
    try:
        series = series.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("y must contain numeric values") from exc
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("y must not contain missing or infinite values")
    if not series.index.is_unique:
        raise ValueError("y index must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError("y index must be monotonically increasing")
    return series


def validate_levels(level: float | Iterable[float]) -> tuple[float, ...]:
    values = (float(level),) if np.isscalar(level) else tuple(float(x) for x in level)
    if not values or any(x <= 0 or x >= 100 for x in values):
        raise ValueError("level values must be between 0 and 100")
    return tuple(dict.fromkeys(values))


def future_index(index: pd.Index, horizon: int) -> pd.Index:
    """Extend common pandas indexes without silently guessing irregular frequencies."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[-1] + 1, periods=horizon, freq=index.freq)
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or index.inferred_freq
        if freq is None:
            raise ValueError("DatetimeIndex must have a regular frequency")
        return pd.date_range(index[-1], periods=horizon + 1, freq=freq)[1:]
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(
            index[-1] + index.step, index[-1] + index.step * (horizon + 1), index.step
        )
    if np.issubdtype(index.dtype, np.number):
        step = index[-1] - index[-2] if len(index) > 1 else 1
        return pd.Index(index[-1] + step * np.arange(1, horizon + 1))
    return pd.RangeIndex(len(index), len(index) + horizon)
