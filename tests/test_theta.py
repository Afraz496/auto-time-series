import numpy as np
import pandas as pd
import pytest

from omnicast import ThetaForecaster


def trending_series(n=24):
    rng = np.random.default_rng(1)
    return pd.Series(
        10 + 0.5 * np.arange(n) + rng.normal(0, 0.3, n),
        index=pd.period_range("2020-01", periods=n, freq="M"),
    )


def seasonal_series(n=36, period=12):
    rng = np.random.default_rng(2)
    trend = 50 + 0.4 * np.arange(n)
    season = 1 + 0.2 * np.sin(2 * np.pi * np.arange(n) / period)
    return pd.Series(
        trend * season + rng.normal(0, 0.5, n),
        index=pd.period_range("2018-01", periods=n, freq="M"),
    )


def test_theta_forecast_tracks_trend():
    y = trending_series()
    forecast = ThetaForecaster().fit(y).predict(6, level=[80, 95])
    assert len(forecast.mean) == 6
    assert np.isfinite(forecast.mean).all()
    assert (forecast.lower[80] <= forecast.mean).all()
    assert (forecast.upper[95] >= forecast.upper[80]).all()
    # Should broadly continue the upward trend rather than flatten out.
    assert forecast.mean.iloc[-1] > forecast.mean.iloc[0]


def test_theta_fitted_values_align_with_input_index():
    y = trending_series()
    model = ThetaForecaster().fit(y)
    assert model.fitted_values_.index.equals(y.index)
    assert np.isfinite(model.fitted_values_.to_numpy()).all()


def test_seasonal_theta_reseasonalizes():
    y = seasonal_series()
    forecast = ThetaForecaster(seasonal_period=12).fit(y).predict(12)
    assert len(forecast.mean) == 12
    assert np.isfinite(forecast.mean).all()


def test_seasonal_theta_rejects_nonpositive_values():
    y = pd.Series(np.arange(-5, 19, dtype=float))
    with pytest.raises(ValueError):
        ThetaForecaster(seasonal_period=6).fit(y)


def test_theta_requires_minimum_observations():
    with pytest.raises(ValueError):
        ThetaForecaster().fit(pd.Series([1.0, 2.0, 3.0]))
