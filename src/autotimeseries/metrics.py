"""Forecast accuracy metrics."""

import numpy as np


def mae(actual, predicted) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(actual) - np.asarray(predicted)))))


def mape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if np.any(actual == 0):
        raise ValueError("MAPE is undefined when actual values contain zero")
    return float(100 * np.mean(np.abs((actual - predicted) / actual)))


def smape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    denominator = np.abs(actual) + np.abs(predicted)
    ratio = np.divide(
        2 * np.abs(actual - predicted),
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return float(100 * np.mean(ratio))


METRICS = {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}
