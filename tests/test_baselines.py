import numpy as np
import pandas as pd
import pytest

from autotimeseries import DriftForecaster, MeanForecaster, NaiveForecaster, SeasonalNaiveForecaster


@pytest.mark.parametrize(
    "model", [NaiveForecaster(), MeanForecaster(), DriftForecaster(), SeasonalNaiveForecaster(4)]
)
def test_baselines_return_labelled_intervals(model):
    y = pd.Series(np.arange(16.0), index=pd.period_range("2020-01", periods=16, freq="M"))
    forecast = model.fit(y).predict(3, level=[80, 95])
    assert len(forecast.mean) == 3
    assert forecast.mean.index[0] == pd.Period("2021-05", freq="M")
    assert (forecast.lower[80] <= forecast.mean).all()
    assert (forecast.upper[95] >= forecast.upper[80]).all()
    assert model.fitted_values_.index.equals(y.index)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError):
        NaiveForecaster().predict(2)
