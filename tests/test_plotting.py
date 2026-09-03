import matplotlib

matplotlib.use("Agg")  # headless backend for testing

import matplotlib.axes
import numpy as np
import pandas as pd

from omnicast.auto import AutoForecaster
from omnicast.evaluation import backtest
from omnicast.models import DriftForecaster, MeanForecaster, NaiveForecaster
from omnicast.plotting import plot_backtest, plot_metric_by_horizon
from omnicast.result import BacktestResult


def test_plot_metric_by_horizon(tmp_path):
    df = pd.DataFrame(
        {
            "model": ["Naive", "Naive", "Theta", "Theta"],
            "horizon": [1, 2, 1, 2],
            "rmse": [10.5, 12.1, 8.4, 9.2],
        }
    )
    save_file = tmp_path / "plot_horizon.png"

    ax = plot_metric_by_horizon(
        df, "rmse", title="Test", ylabel="RMSE", target_val=10.0, save_path=str(save_file)
    )

    assert isinstance(ax, matplotlib.axes.Axes)
    assert save_file.stat().st_size > 0


def test_forecast_result_infers_observed(tmp_path):
    y = pd.Series(np.arange(24.0), index=pd.period_range("2024-01", periods=24, freq="M"))
    fc = NaiveForecaster().fit(y).predict(horizon=4, level=[80, 95])

    assert fc.observed is not None and fc.observed.equals(y)  # forecaster stashes the series
    save_file = tmp_path / "traj.png"
    ax = fc.plot(save_path=str(save_file))  # no observed= argument
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("Observed" in t for t in labels)
    assert labels.index("80% interval") < labels.index("95% interval")  # nested, narrow first
    assert save_file.stat().st_size > 0

    bt = backtest(NaiveForecaster(), y, horizon=2, initial=18)
    assert bt.observed is not None and bt.observed.equals(y)
    assert any("Observed" in t.get_text() for t in bt.plot().get_legend().get_texts())


def test_plot_backtest_predictions(tmp_path):
    y = pd.Series(np.arange(20.0), index=pd.period_range("2024-01", periods=20, freq="M"))
    bt = backtest(NaiveForecaster(), y, horizon=2, initial=14, metric="rmse")

    assert isinstance(bt, BacktestResult)
    assert not bt.predictions.empty
    assert {"predicted", "cutoff", "target_date"} <= set(bt.predictions.columns)

    summary = bt.summary()
    assert summary["count"] == len(bt)
    assert "mean" in summary

    save_file = tmp_path / "plot_backtest.png"
    ax = plot_backtest(bt, observed=y, title="Backtest Folds", save_path=str(save_file))
    assert isinstance(ax, matplotlib.axes.Axes)
    assert save_file.stat().st_size > 0

    method_ax = bt.plot(observed=y)
    assert method_ax.get_title() == "Backtest Predictions (NaiveForecaster)"

    # observed omitted: falls back to result.observed, then to predictions['actual']
    def has_observed_line(a):
        return any("Observed" in t.get_text() for t in a.get_legend().get_texts())

    assert has_observed_line(plot_backtest(bt))
    assert has_observed_line(plot_backtest(bt.predictions))  # raw DataFrame, no .observed


def test_auto_forecaster_plot_all():
    y = pd.Series(np.arange(24.0), index=pd.period_range("2024-01", periods=24, freq="M"))
    auto = AutoForecaster(
        models=[NaiveForecaster(), MeanForecaster(), DriftForecaster()],
        validation_horizon=3,
    ).fit(y)

    ax = auto.plot_all(horizon=6)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "DriftForecaster forecast" in labels
    assert "MeanForecaster forecast" in labels
