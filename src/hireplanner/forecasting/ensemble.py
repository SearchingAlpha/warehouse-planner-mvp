"""Ensemble / model selection logic for combining forecasters."""

import numpy as np
import pandas as pd
from typing import Optional


def select_best_model(
    models_results: dict[str, dict[str, np.ndarray]],
    actuals: Optional[np.ndarray] = None,
) -> str:
    """Select the best model based on recent performance.

    If actuals are provided, select based on lowest WAPE on historical validation.
    Otherwise, default to 'lightgbm' if available, else first available.
    """
    if actuals is not None and len(actuals) > 0:
        from hireplanner.metrics.evaluation import wape

        best_name = None
        best_wape = float("inf")
        for name, result in models_results.items():
            pred = result["p50"][: len(actuals)]
            w = wape(actuals[: len(pred)], pred)
            if w < best_wape:
                best_wape = w
                best_name = name
        return best_name

    # Default priority
    if "lightgbm" in models_results:
        return "lightgbm"
    return next(iter(models_results))


def blend_forecasts(
    models_results: dict[str, dict[str, np.ndarray]],
    weights: Optional[dict[str, float]] = None,
) -> dict[str, np.ndarray]:
    """Blend multiple model forecasts using weighted average.

    Args:
        models_results: dict of model_name -> {"p10": arr, "p50": arr, "p90": arr}
        weights: Optional dict of model_name -> weight. If None, equal weights.

    Returns:
        Blended forecast dict with p10, p50, p90.
    """
    if len(models_results) == 1:
        return next(iter(models_results.values()))

    names = list(models_results.keys())
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}

    # Normalize weights
    total_w = sum(weights[n] for n in names)

    result = {}
    for key in ("p10", "p50", "p90"):
        blended = np.zeros_like(models_results[names[0]][key])
        for name in names:
            blended += models_results[name][key] * (weights[name] / total_w)
        result[key] = np.maximum(0, blended)

    # Carry dates if available
    for name in names:
        if "dates" in models_results[name]:
            result["dates"] = models_results[name]["dates"]
            break

    return result


def build_forecast_df(
    forecast_result: dict[str, np.ndarray],
    dates: pd.DatetimeIndex | np.ndarray,
    flow: str,
) -> pd.DataFrame:
    """Convert forecast result dict to standard DataFrame format.

    Returns DataFrame with columns: date, flow, forecast_p50, forecast_p10, forecast_p90
    """
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "flow": flow,
        "forecast_p50": forecast_result["p50"],
        "forecast_p10": forecast_result["p10"],
        "forecast_p90": forecast_result["p90"],
    })
