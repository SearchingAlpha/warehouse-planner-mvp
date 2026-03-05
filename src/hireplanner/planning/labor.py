import math

import numpy as np
import pandas as pd


def calculate_headcount(
    forecast_volume: pd.Series | np.ndarray,
    productivity: float,
    hours_per_shift: float,
    overhead_buffer: float = 0.15,
) -> np.ndarray:
    """Calculate daily headcount from forecast volume.

    headcount = ceil(volume / (productivity * hours_per_shift) * (1 + overhead_buffer))
    Returns array of integers (whole workers).
    """
    volume = np.asarray(forecast_volume, dtype=float)
    raw = volume / (productivity * hours_per_shift) * (1 + overhead_buffer)
    return np.array([max(0, math.ceil(v)) for v in raw])


def calculate_daily_hours(
    headcount: np.ndarray,
    hours_per_shift: float,
) -> np.ndarray:
    """Calculate total labor hours per day."""
    return np.asarray(headcount, dtype=float) * hours_per_shift


def build_headcount_plan(
    forecast_df: pd.DataFrame,
    config,  # ClientConfig
) -> pd.DataFrame:
    """Build a complete headcount plan DataFrame from forecast and config.

    forecast_df must have columns: date, flow, forecast_p50
    Returns DataFrame with columns: date, hc_outbound, hc_inbound, hc_total
    """
    dates = forecast_df[forecast_df["flow"] == forecast_df["flow"].iloc[0]]["date"].values

    result = pd.DataFrame({"date": dates})
    result["hc_outbound"] = 0
    result["hc_inbound"] = 0

    if "outbound" in config.active_flows:
        ob = forecast_df[forecast_df["flow"] == "outbound"]["forecast_p50"].values
        result["hc_outbound"] = calculate_headcount(
            ob, config.productivity_outbound, config.hours_per_shift, config.overhead_buffer
        )

    if "inbound" in config.active_flows:
        ib = forecast_df[forecast_df["flow"] == "inbound"]["forecast_p50"].values
        result["hc_inbound"] = calculate_headcount(
            ib, config.productivity_inbound, config.hours_per_shift, config.overhead_buffer
        )

    result["hc_total"] = result["hc_outbound"] + result["hc_inbound"]
    return result
