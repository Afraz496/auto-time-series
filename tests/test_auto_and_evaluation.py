import numpy as np
import pandas as pd

from autotimeseries import (
    AutoForecaster,
    DriftForecaster,
    MeanForecaster,
    NaiveForecaster,
    backtest,
)


def test_backtest_has_temporal_folds():
    result = backtest(NaiveForecaster(), pd.Series(np.arange(20.0)), horizon=2, initial=12)
    assert list(result.columns) == ["cutoff", "score", "n_train"]
    assert len(result) == 7


def test_auto_forecaster_selects_and_refits():
    y = pd.Series(np.arange(20.0))
    model = AutoForecaster(models=[MeanForecaster(), NaiveForecaster(), DriftForecaster()])
    forecast = model.fit(y).predict(3)
    assert model.leaderboard_.iloc[0]["model"] == "DriftForecaster"
    assert np.allclose(forecast.mean, [20, 21, 22])


def test_auto_forecaster_does_not_mutate_caller_supplied_models():
    y = pd.Series(np.arange(20.0))
    user_models = [MeanForecaster(), NaiveForecaster()]
    model = AutoForecaster(models=user_models, seasonal_period=4)
    model.fit(y).fit(y)  # fit twice: a mutating bug would grow the list each call
    assert len(user_models) == 2
    assert len(model.leaderboard_) == 3
