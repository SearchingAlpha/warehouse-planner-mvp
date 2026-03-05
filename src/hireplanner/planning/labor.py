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
    Returns DataFrame with columns:
        date,
        hc_outbound_recommended, hc_outbound_actual,
        hc_inbound_recommended, hc_inbound_actual,
        hc_total_recommended, hc_total_actual,
        hc_outbound, hc_inbound, hc_total (aliases for recommended)
    """
    from hireplanner.planning.backlog import calculate_recommended_capacity

    dates = forecast_df[forecast_df["flow"] == forecast_df["flow"].iloc[0]]["date"].values

    result = pd.DataFrame({"date": dates})
    result["hc_outbound_recommended"] = 0
    result["hc_outbound_actual"] = 0
    result["hc_inbound_recommended"] = 0
    result["hc_inbound_actual"] = 0

    if "outbound" in config.active_flows:
        ob = forecast_df[forecast_df["flow"] == "outbound"]["forecast_p50"].values
        initial = config.initial_backlog_outbound
        rec_hc, _ = calculate_recommended_capacity(
            ob, initial,
            config.target_backlog_ratio,
            config.backlog_threshold_critical,
            config.productivity_outbound,
            config.hours_per_shift,
        )
        result["hc_outbound_recommended"] = rec_hc
        result["hc_outbound_actual"] = config.current_staffing_outbound

    if "inbound" in config.active_flows:
        ib = forecast_df[forecast_df["flow"] == "inbound"]["forecast_p50"].values
        initial = config.initial_backlog_inbound
        rec_hc, _ = calculate_recommended_capacity(
            ib, initial,
            config.target_backlog_ratio,
            config.backlog_threshold_critical,
            config.productivity_inbound,
            config.hours_per_shift,
        )
        result["hc_inbound_recommended"] = rec_hc
        result["hc_inbound_actual"] = config.current_staffing_inbound

    result["hc_total_recommended"] = (
        result["hc_outbound_recommended"] + result["hc_inbound_recommended"]
    )
    result["hc_total_actual"] = (
        result["hc_outbound_actual"] + result["hc_inbound_actual"]
    )

    # Backward-compat aliases (point to recommended)
    result["hc_outbound"] = result["hc_outbound_recommended"]
    result["hc_inbound"] = result["hc_inbound_recommended"]
    result["hc_total"] = result["hc_total_recommended"]

    return result
