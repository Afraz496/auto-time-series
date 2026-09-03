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

__all__ = ["plot_backtest", "plot_metric_by_horizon"]


def _to_timestamp_safe(s: pd.Series | pd.Index) -> pd.Series | pd.Index:
    """Convert a PeriodIndex / series of Periods to timestamps for matplotlib."""
    if isinstance(s, pd.Index):
        return s.to_timestamp() if hasattr(s, "to_timestamp") else pd.to_datetime(s)
    if len(s) and hasattr(s.iloc[0], "to_timestamp"):
        return pd.Series([x.to_timestamp() for x in s], index=s.index)
    return pd.to_datetime(s)


def _anchor_before(observed: pd.Series | None, first_ts) -> tuple | None:
    """Last observed (timestamp, value) strictly before ``first_ts``.

    Joins a forecast line to the point it was made from, even when ``observed``
    runs past the forecast start (train/test overlay).
    """
    if observed is None or not len(observed):
        return None
    ts = _to_timestamp_safe(observed.index)
    mask = ts < first_ts
    if not mask.any():
        return None
    return ts[mask][-1], float(observed.to_numpy()[mask][-1])


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
    forecasts: ForecastResult | dict[str, ForecastResult],
    observed: pd.Series | None = None,
    title: str = "Forecast Trajectory",
    intervals: bool = True,
    figsize: tuple[float, float] = (10, 5),
    ax: Axes | None = None,
    save_path: str | None = None,
) -> Axes:
    """Plot one or several forecast trajectories over the observed history.

    Shared implementation behind ``ForecastResult.plot()`` and
    ``AutoForecaster.plot_all()`` -- not part of the public API; call those.

    Parameters
    ----------
    forecasts : ForecastResult or dict[str, ForecastResult]
        A single forecast, or ``{label: forecast}`` to overlay several on one axes.
    observed : pd.Series, optional
        Historical observations, indexed by period or timestamp.
    title : str, default "Forecast Trajectory"
    intervals : bool, default True
        Shade every prediction interval each forecast carries (nested). Turn off
        for a cleaner multi-model comparison.
    figsize : tuple, default (10, 5)
    ax : matplotlib Axes, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if not isinstance(forecasts, dict):
        forecasts = {forecasts.model_name: forecasts}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ylabel = _plot_observed(ax, observed)
    palette = sns.color_palette(n_colors=len(forecasts))

    single = len(forecasts) == 1
    for (name, fc), color in zip(forecasts.items(), palette):
        df = fc.to_frame()
        x = _to_timestamp_safe(df.index)
        anchor = _anchor_before(observed, x[0])  # join to the last obs before the forecast
        connect = anchor is not None
        xc = x.insert(0, anchor[0]) if connect else x

        mean_y = df["mean"].to_numpy()
        if connect:
            mean_y = [anchor[1], *mean_y]
        # line first so the legend reads: forecast, then intervals narrow -> wide
        sns.lineplot(
            x=xc,
            y=mean_y,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"{name} forecast",
            ax=ax,
            zorder=3,
        )
        if intervals:
            for lvl in sorted(fc.lower):
                lo = df[f"lower_{lvl:g}"].to_numpy()
                hi = df[f"upper_{lvl:g}"].to_numpy()
                if connect:
                    lo = [anchor[1], *lo]
                    hi = [anchor[1], *hi]
                ax.fill_between(
                    xc,
                    lo,
                    hi,
                    color=color,
                    alpha=0.15,
                    zorder=-lvl,  # wider band sits behind the narrower one
                    label=f"{lvl:g}% interval" if single else None,
                )

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
        History to draw the predictions against. Defaults to ``result.observed``
        (a ``BacktestResult`` always carries it), then to the ``actual`` column
        of the predictions.
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

    if observed is None:
        observed = getattr(result, "observed", None)
    if observed is None and "actual" in preds.columns:  # realized values per target date
        actuals = (
            preds.dropna(subset=["actual"])
            .drop_duplicates("target_date")
            .set_index("target_date")["actual"]
            .sort_index()
        )
        observed = actuals if len(actuals) else None

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.set_theme(style="whitegrid")

    ylabel = _plot_observed(ax, observed)
    obs_ts = None
    if observed is not None and len(observed):
        obs_ts = pd.Series(
            observed.to_numpy(), index=_to_timestamp_safe(observed.index)
        ).sort_index()

    lower_col = next((c for c in preds.columns if c.startswith("lower")), None)
    upper_col = next((c for c in preds.columns if c.startswith("upper")), None)
    for i, cutoff in enumerate(sorted(preds["cutoff"].unique())):
        sub = preds[preds["cutoff"] == cutoff].sort_values("target_date")
        td = list(sub["target_date"])
        pred = list(sub["predicted"])
        band = show_intervals and lower_col and upper_col
        lo = list(sub[lower_col]) if band else None
        hi = list(sub[upper_col]) if band else None

        if obs_ts is not None:  # join each fold to the observation at its cutoff
            av = obs_ts.asof(pd.Timestamp(cutoff))
            if pd.notna(av):
                td = [pd.Timestamp(cutoff), *td]
                pred = [av, *pred]
                if band:
                    lo = [av, *lo]
                    hi = [av, *hi]

        if band:
            ax.fill_between(
                td,
                lo,
                hi,
                color="#1f77b4",
                alpha=0.15,
                label="Prediction interval" if i == 0 else None,
            )
        sns.lineplot(
            x=td,
            y=pred,
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
