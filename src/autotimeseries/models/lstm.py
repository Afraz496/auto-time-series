"""Optional PyTorch-backed neural forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseForecaster

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only when torch is missing
    torch = None
    nn = None


if nn is not None:

    class _LSTMNet(nn.Module):
        def __init__(self, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])


class LSTMForecaster(BaseForecaster):
    """Autoregressive LSTM forecaster wrapping `torch.nn.LSTM`.

    Not an R port -- this wraps a major Python deep-learning module the same
    way `ETSForecaster`/`ARIMAForecaster` wrap `statsmodels`. Requires the
    optional `torch` extra (`pip install auto-time-series[torch]`); raises a
    clear `ImportError` naming the missing package if `torch` is absent, and
    the module still imports cleanly without it.

    The series is standardized (zero mean, unit variance, computed from
    training data only) and reframed as a sliding-window supervised
    regression: `lookback` consecutive points predict the next one. A
    single-layer (by default) LSTM is trained with Adam/MSE, full-batch, for
    `epochs` iterations -- appropriate for the short series typical of this
    package, not for large-scale training. Multi-step forecasts are produced
    autoregressively: each predicted point is fed back in as the most recent
    observation of the next window.

    Prediction intervals use the same residual-variance random-walk scaling
    (`sqrt(sigma2 * h)`) as `NaiveForecaster` and `ThetaForecaster` -- an
    approximation, since the network has no closed-form predictive variance.

    Not included in `AutoForecaster`'s default candidate list: it is
    optional-dependency and materially slower to backtest than the built-in
    statistical models. Pass it explicitly via `AutoForecaster(models=[...])`
    to include it.

    Minimum sample size: `lookback + 2` observations. Training is not
    deterministic across torch versions/hardware even with a fixed `seed`;
    do not rely on exact reproducibility across environments.

    Examples
    --------
    >>> import pandas as pd
    >>> from autotimeseries import LSTMForecaster
    >>> y = pd.Series([10.0, 12.0, 11.0, 13.0, 15.0, 14.0, 16.0, 15.0])
    >>> model = LSTMForecaster(lookback=2, hidden_size=4, epochs=30, seed=0).fit(y)
    >>> forecast = model.predict(horizon=2)  # values vary slightly by platform

    Notes
    -----
    .. list-table:: When to use this model
       :header-rows: 1
       :widths: 25 75

       * - Best for
         - Exploring a nonlinear, learned alternative once the built-in
           statistical models have been tried; short series only
       * - Avoid when
         - You need a fast default, exact reproducibility across
           machines, or don't want the optional ``torch`` dependency --
           never included in :class:`~autotimeseries.AutoForecaster`'s
           default candidates for these reasons
       * - Handles trend
         - Implicitly, via the sliding-window autoregression
       * - Handles seasonality
         - Only if ``lookback`` spans a full cycle; no explicit seasonal
           component
       * - Extra dependencies
         - ``torch`` (``pip install auto-time-series[torch]``)
       * - Min. observations
         - ``lookback + 2``
    """

    def __init__(
        self,
        lookback: int = 12,
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 200,
        learning_rate: float = 1e-2,
        dropout: float = 0.0,
        seed: int = 0,
    ):
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed

    def _fit(self, y: pd.Series) -> None:
        if torch is None:
            raise ImportError(
                "LSTMForecaster requires the optional 'torch' dependency; "
                "install with `pip install auto-time-series[torch]`."
            )
        n = len(y)
        if n < self.lookback + 2:
            raise ValueError(
                f"LSTMForecaster requires at least {self.lookback + 2} observations "
                f"for lookback={self.lookback}"
            )

        torch.manual_seed(self.seed)
        values = y.to_numpy(dtype=float)
        self.mean_ = float(values.mean())
        self.std_ = float(values.std()) or 1.0
        scaled = (values - self.mean_) / self.std_

        windows = np.stack([scaled[i : i + self.lookback] for i in range(n - self.lookback)])
        targets = scaled[self.lookback :]
        x = torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
        target = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

        model = _LSTMNet(self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(x), target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            fitted_scaled = model(x).squeeze(-1).numpy()
        self._model = model
        self._n = n
        self._last_window_ = scaled[-self.lookback :].copy()
        self._fitted_scaled_ = fitted_scaled

    def _fitted_values(self) -> np.ndarray:
        fitted = np.full(self._n, np.nan)
        fitted[self.lookback :] = self._fitted_scaled_ * self.std_ + self.mean_
        return fitted

    def _forecast(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        window = self._last_window_.copy()
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(window, dtype=torch.float32).reshape(1, self.lookback, 1)
                next_scaled = self._model(x).item()
                preds.append(next_scaled)
                window = np.r_[window[1:], next_scaled]
        mean = np.array(preds) * self.std_ + self.mean_
        steps = np.arange(1, horizon + 1)
        se = np.sqrt(self.sigma2_ * steps)
        return mean, se
