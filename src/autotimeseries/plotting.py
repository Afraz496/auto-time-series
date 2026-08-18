"""Seaborn plots for forecasts, backtests, and metric-by-horizon comparisons.

Every function returns the matplotlib ``Axes`` it drew on. matplotlib and
seaborn are hard dependencies, so there is no optional-import dance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .result import BacktestResult, ForecastResult

__all__ = ["plot_backtest", "plot_forecast_trajectories", "plot_metric_by_horizon"]


def _to_timestamp_safe(s: pd.Series | pd.Index) -> pd.Series | pd.Index:
    """Convert a PeriodIndex / series of Periods to timestamps for matplotlib."""
    if isinstance(s, pd.Index):
        return s.to_timestamp() if hasattr(s, "to_timestamp") else pd.to_datetime(s)
    if len(s) and hasattr(s.iloc[0], "to_timestamp"):
        return pd.Series([x.to_timestamp() for x in s], index=s.index)
    return pd.to_datetime(s)


def _plot_observed(ax: Axes, observed: pd.Series | None) -> str:
    """Draw the observed history (if any); return a y-axis label."""
    if observed is None:
        return "Value"
    label = str(observed.name or "value").replace("_", " ").title()
    sns.lineplot(
        x=_to_timestamp_safe(observed.index),
        y=observed.to_numpy(),
        color="#222222",
        label=f"Observed {label}",
        linewidth=1.6,
        ax=ax,
    )
    return label


def plot_metric_by_horizon(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str = "Metric by Forecast Horizon",
    ylabel: str | None = None,
    colors: dict[str, str] | None = None,
    horizons: list[int] | None = None,
    target_val: float | None = None,
    is_percentage: bool = False,
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    save_path: str | None = None,
) -> Axes:
    """Plot a metric across forecast horizons, one line per model.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Long-format table with columns ``model``, ``horizon``, and ``metric_col``.
    metric_col : str
        Metric column to plot on the y-axis.
    title, ylabel : str, optional
        Title and y-axis label (``ylabel`` defaults to ``metric_col``).
    colors : dict, optional
        Mapping of model name to line colour.
    horizons : list of int, optional
        x-axis ticks. Defaults to the sorted unique horizons in the data.
    target_val : float, optional
        Reference value drawn as a horizontal dashed line.
    is_percentage : bool, default False
        Multiply values and ``target_val`` by 100 before plotting.
    figsize : tuple, default (10, 5)
        Figure size when a new figure is created.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is created when omitted.
    save_path : str, optional
        If given, save the figure here at 300 dpi.

    Returns
    -------
    matplotlib.axes.Axes
    """
    missing = {"model", "horizon", metric_col} - set(metrics_df.columns)
    if missing:
        raise KeyError(f"metrics_df is missing columns: {sorted(missing)}")

    df_plot = metrics_df.copy()
    if is_percentage:
        df_plot[metric_col] = df_plot[metric_col] * 100.0

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df_plot,
        x="horizon",
        y=metric_col,
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        palette=colors or None,
        linewidth=2.0,
        markersize=7,
        ax=ax,
    )

    if target_val is not None:
        ref = target_val * 100.0 if is_percentage else target_val
        ax.axhline(ref, color="black", linestyle="--", alpha=0.7, label=f"Target ({ref:.0f})")

    ax.set_xticks(horizons or sorted(df_plot["horizon"].unique()))
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel(ylabel or metric_col, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=8)
    ax.figure.tight_layout()
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_forecast_trajectories(
    observed: pd.Series | None,
    forecast: ForecastResult,
    title: str = "Forecast Trajectory",
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    save_path: str | None = None,
) -> Axes:
    """Plot a forecast trajectory with its tightest prediction interval.

    Parameters
    ----------
    observed : pd.Series, optional
        Historical observations, indexed by period or timestamp.
    forecast : ForecastResult
        The forecast to draw.
    title : str, default "Forecast Trajectory"
    figsize : tuple, default (10, 5)
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ylabel = _plot_observed(ax, observed)

    df = forecast.to_frame()
    x = _to_timestamp_safe(df.index)
    ax.axvline(x[0], color="grey", linestyle="--", alpha=0.5)

    level = min(forecast.lower)
    ax.fill_between(
        x,
        df[f"lower_{level:g}"],
        df[f"upper_{level:g}"],
        color="#d95f02",
        alpha=0.2,
        label=f"{level:g}% interval",
    )
    sns.lineplot(x=x, y=df["mean"].to_numpy(), color="#d95f02", linewidth=2.0, label="Forecast", ax=ax)

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.figure.tight_layout()
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax


def plot_backtest(
    result: BacktestResult | pd.DataFrame,
    observed: pd.Series | None = None,
    title: str = "Backtest Out-of-Sample Predictions",
    show_intervals: bool = True,
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    save_path: str | None = None,
) -> Axes:
    """Plot rolling-origin backtest predictions against observations.

    Parameters
    ----------
    result : BacktestResult or pd.DataFrame
        A ``BacktestResult`` (its ``.predictions`` are used) or the predictions
        table directly: columns ``cutoff``, ``target_date``, ``predicted``, and
        optional ``lower_*`` / ``upper_*``.
    observed : pd.Series, optional
        Historical observations.
    title : str, default "Backtest Out-of-Sample Predictions"
    show_intervals : bool, default True
        Draw the per-fold interval band when interval columns are present.
    figsize : tuple, default (10, 5)
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    preds = getattr(result, "predictions", result)
    if preds is None or preds.empty:
        raise ValueError("backtest result has no predictions to plot")
    model_name = getattr(result, "model_name", "model")

    preds = preds.copy()
    preds["cutoff"] = _to_timestamp_safe(preds["cutoff"])
    preds["target_date"] = _to_timestamp_safe(preds["target_date"])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ylabel = _plot_observed(ax, observed)

    lower_col = next((c for c in preds.columns if c.startswith("lower")), None)
    upper_col = next((c for c in preds.columns if c.startswith("upper")), None)
    for i, cutoff in enumerate(sorted(preds["cutoff"].unique())):
        sub = preds[preds["cutoff"] == cutoff].sort_values("target_date")
        ax.axvline(cutoff, color="grey", linestyle=":", alpha=0.45)
        if show_intervals and lower_col and upper_col:
            ax.fill_between(
                sub["target_date"],
                sub[lower_col],
                sub[upper_col],
                color="#1f77b4",
                alpha=0.15,
                label="Prediction interval" if i == 0 else None,
            )
        sns.lineplot(
            data=sub,
            x="target_date",
            y="predicted",
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=5,
            ax=ax,
            label=f"Backtest folds ({model_name})" if i == 0 else None,
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.figure.tight_layout()
    if save_path:
        ax.figure.savefig(save_path, dpi=300, bbox_inches="tight")
    return ax
