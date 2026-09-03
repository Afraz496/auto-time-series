import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from omnicast import LSTMForecaster


def trending_series(n=30):
    rng = np.random.default_rng(3)
    return pd.Series(
        10 + 0.4 * np.arange(n) + rng.normal(0, 0.2, n),
        index=pd.period_range("2020-01", periods=n, freq="M"),
    )


def test_lstm_forecast_shape_and_finiteness():
    y = trending_series()
    model = LSTMForecaster(lookback=6, hidden_size=8, epochs=20, seed=0)
    forecast = model.fit(y).predict(4, level=[80, 95])
    assert len(forecast.mean) == 4
    assert np.isfinite(forecast.mean).all()
    assert (forecast.lower[80] <= forecast.mean).all()
    assert (forecast.upper[95] >= forecast.upper[80]).all()


def test_lstm_fitted_values_align_with_input_index_and_lookback_gap():
    y = trending_series()
    model = LSTMForecaster(lookback=6, hidden_size=8, epochs=20, seed=0)
    model.fit(y)
    assert model.fitted_values_.index.equals(y.index)
    assert model.fitted_values_.iloc[:6].isna().all()
    assert np.isfinite(model.fitted_values_.iloc[6:].to_numpy()).all()


def test_lstm_requires_minimum_observations():
    with pytest.raises(ValueError):
        LSTMForecaster(lookback=12).fit(pd.Series(np.arange(10.0)))
