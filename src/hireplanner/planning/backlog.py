"""Backlog calculation engine for warehouse planning."""
import math
import numpy as np
import pandas as pd


def calculate_daily_backlog(
    forecast_demand: np.ndarray | pd.Series,
    capacity_per_day: np.ndarray | pd.Series,
    initial_backlog: float,
) -> pd.DataFrame:
    """Calculate daily backlog evolution.

    Core equation: end_backlog = max(0, beg_backlog - capacity + new_demand)

    Returns DataFrame with columns: beg_backlog, new_demand, capacity, end_backlog
    """
    demand = np.asarray(forecast_demand, dtype=float)
    capacity = np.asarray(capacity_per_day, dtype=float)
    n = len(demand)

    beg = np.zeros(n)
    end = np.zeros(n)

    for i in range(n):
        beg[i] = initial_backlog if i == 0 else end[i - 1]
        end[i] = max(0.0, beg[i] - capacity[i] + demand[i])

    return pd.DataFrame({
        "beg_backlog": beg,
        "new_demand": demand,
        "capacity": capacity,
        "end_backlog": end,
    })


def calculate_daily_capacity(
    headcount: np.ndarray,
    productivity: float,
    hours_per_shift: float,
) -> np.ndarray:
    """Convert headcount to daily processing capacity (units/day)."""
    return np.asarray(headcount, dtype=float) * productivity * hours_per_shift


def calculate_days_of_backlog(
    end_backlog: np.ndarray | pd.Series,
    capacity: np.ndarray | pd.Series,
    window: int = 7,
) -> np.ndarray:
    """Calculate days of backlog for each day.

    days_of_backlog[i] = end_backlog[i] / mean(capacity[i+1:i+1+window])
    At end of horizon, uses remaining available days.
    Returns 0 if capacity is 0.
    """
    backlog = np.asarray(end_backlog, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    n = len(backlog)
    result = np.zeros(n)

    for i in range(n):
        # Look at next `window` days of capacity
        start = i + 1
        end_idx = min(start + window, n)
        if start >= n:
            # Last day: use current day's capacity
            avg_cap = cap[i] if cap[i] > 0 else 1.0
        else:
            future_cap = cap[start:end_idx]
            avg_cap = np.mean(future_cap) if len(future_cap) > 0 and np.mean(future_cap) > 0 else 1.0
        result[i] = backlog[i] / avg_cap

    return result


def calculate_flow_backlog(
    config,  # ClientConfig
    forecast_df: pd.DataFrame,
    flow: str,
) -> pd.DataFrame:
    """Calculate complete backlog projection for a single flow.

    Args:
        config: ClientConfig with productivity rates and initial backlogs
        forecast_df: DataFrame with columns date, flow, forecast_p50
        flow: "outbound" or "inbound"

    Returns:
        DataFrame with columns: date, beg_backlog, new_demand, capacity,
                                end_backlog, days_of_backlog
    """
    from hireplanner.planning.labor import calculate_headcount

    flow_data = forecast_df[forecast_df["flow"] == flow].copy().reset_index(drop=True)
    demand = flow_data["forecast_p50"].values

    productivity = getattr(config, f"productivity_{flow}")
    initial = getattr(config, f"initial_backlog_{flow}")

    # Calculate recommended headcount for this flow
    hc = calculate_headcount(demand, productivity, config.hours_per_shift, config.overhead_buffer)

    # Convert headcount to capacity
    cap = calculate_daily_capacity(hc, productivity, config.hours_per_shift)

    # Calculate backlog
    backlog_df = calculate_daily_backlog(demand, cap, initial)
    backlog_df.insert(0, "date", flow_data["date"].values)

    # Calculate days of backlog
    backlog_df["days_of_backlog"] = calculate_days_of_backlog(
        backlog_df["end_backlog"].values, cap
    )

    return backlog_df


def calculate_all_backlogs(
    config,  # ClientConfig
    forecast_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate backlog projections for all active flows.

    Returns dict mapping flow name to backlog DataFrame.
    """
    result = {}
    for flow in config.active_flows:
        result[flow] = calculate_flow_backlog(config, forecast_df, flow)
    return result
