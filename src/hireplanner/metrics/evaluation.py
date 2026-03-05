import numpy as np
import pandas as pd


def wape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Weighted Absolute Percentage Error. Returns value between 0 and 1+."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    total = np.sum(np.abs(actual))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / total)


def mape(actual, predicted) -> float:
    """Mean Absolute Percentage Error. Filters out zero actuals. Returns 0-1+ scale."""
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mask = actual != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def mae(actual, predicted) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def rmse(actual, predicted) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)) ** 2)))


def evaluate_forecast(actual, predicted) -> dict[str, float]:
    """Calculate all metrics at once. Returns dict with wape, mape, mae, rmse."""
    return {
        "wape": wape(actual, predicted),
        "mape": mape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
    }
