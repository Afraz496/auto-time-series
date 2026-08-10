# Changelog

## 0.1.0

- Common fit/predict API and labelled forecast result objects.
- Naive, seasonal-naive, mean, drift, Theta, ETS, ARIMA, automatic ARIMA, and LSTM models.
- `ThetaForecaster`: the first R port, a compatible reimplementation of `forecast::thetaf`.
- `LSTMForecaster`: the first optional-dependency deep-learning wrapper (`torch`).
- Prediction intervals, residuals, fitted values, model parameters, AIC, and BIC.
- Rolling-origin evaluation, accuracy metrics, and automatic model selection.
- Fixed `AutoForecaster.fit` mutating a caller-supplied `models` list on each call.
- CI workflow running ruff and pytest across Python 3.10-3.12.

