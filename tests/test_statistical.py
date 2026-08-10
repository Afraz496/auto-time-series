import numpy as np
import pandas as pd

from autotimeseries import ARIMAForecaster, AutoARIMAForecaster, ETSForecaster


def series():
    rng = np.random.default_rng(4)
    return pd.Series(10 + 0.2 * np.arange(40) + rng.normal(0, 0.2, 40))


def test_arima_and_ets_forecast():
    for model in [ARIMAForecaster((1, 1, 0)), ETSForecaster()]:
        result = model.fit(series()).predict(4)
        assert len(result.to_frame()) == 4
        assert np.isfinite(result.mean).all()
        assert hasattr(model, "parameter_confidence_intervals_")


def test_small_auto_arima_search():
    model = AutoARIMAForecaster(max_p=1, max_d=1, max_q=1)
    model.fit(series())
    assert len(model.order_) == 3
    assert not model.search_results_.empty
