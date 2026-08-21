import matplotlib

matplotlib.use("Agg")  # headless backend for testing

import matplotlib.axes
import numpy as np
import pandas as pd

from autotimeseries.evaluation import backtest
from autotimeseries.models import NaiveForecaster
from autotimeseries.plotting import (
    plot_backtest,
    plot_forecast_trajectories,
    plot_metric_by_horizon,
)
from autotimeseries.result import BacktestResult, ForecastResult


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


def test_plot_forecast_trajectories(tmp_path):
    idx = pd.period_range("2025-01", periods=12, freq="M")
    observed = pd.Series(np.arange(12), index=idx, name="value")
    idx_fc = pd.period_range("2025-10", periods=3, freq="M")
    forecast = ForecastResult(
        mean=pd.Series([9.5, 10.5, 11.5], index=idx_fc),
        lower={80: pd.Series([8.0, 9.0, 10.0], index=idx_fc)},
        upper={80: pd.Series([11.0, 12.0, 13.0], index=idx_fc)},
        model_name="TestModel",
    )

    save_file = tmp_path / "plot_trajectory.png"
    ax = plot_forecast_trajectories(observed, forecast, title="Traj", save_path=str(save_file))
    assert isinstance(ax, matplotlib.axes.Axes)
    assert save_file.stat().st_size > 0

    method_ax = forecast.plot(observed=observed, title="Method Traj")
    assert method_ax.get_title() == "Method Traj"


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
