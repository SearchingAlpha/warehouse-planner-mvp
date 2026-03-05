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


def calculate_recommended_capacity(
    demand: np.ndarray,
    initial_backlog: float,
    target_backlog_ratio: float,
    backlog_threshold_critical: float,
    productivity: float,
    hours_per_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate recommended headcount and capacity using target backlog.

    The target backlog is computed once from the mean demand, making the
    calculation non-circular.

    Returns:
        (rec_hc, rec_cap) arrays of length len(demand)
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)

    reference_daily_capacity = float(np.mean(demand))
    target_backlog_units = (
        target_backlog_ratio * backlog_threshold_critical * reference_daily_capacity
    )

    rec_hc = np.zeros(n, dtype=int)
    rec_cap = np.zeros(n, dtype=float)

    beg = initial_backlog
    for i in range(n):
        needed = max(0.0, beg + demand[i] - target_backlog_units)
        hc = max(0, math.ceil(needed / (productivity * hours_per_shift)))
        cap = hc * productivity * hours_per_shift
        rec_hc[i] = hc
        rec_cap[i] = cap
        beg = max(0.0, beg + demand[i] - cap)

    return rec_hc, rec_cap


def calculate_actual_capacity(
    n_days: int,
    current_staffing: int,
    productivity: float,
    hours_per_shift: float,
) -> np.ndarray:
    """Calculate constant daily capacity from current staffing.

    Returns array of length n_days with constant capacity.
    """
    cap = current_staffing * productivity * hours_per_shift
    return np.full(n_days, cap, dtype=float)


def calculate_flow_backlog(
    config,  # ClientConfig
    forecast_df: pd.DataFrame,
    flow: str,
) -> pd.DataFrame:
    """Calculate complete backlog projection for a single flow (dual track).

    Returns DataFrame with columns:
        date, beg_backlog, new_demand,
        capacity_recommended, capacity_actual,
        end_backlog_recommended, end_backlog_actual,
        days_of_backlog_recommended, days_of_backlog_actual
    Plus backward-compat aliases: capacity, end_backlog, days_of_backlog (= recommended)
    """
    flow_data = forecast_df[forecast_df["flow"] == flow].copy().reset_index(drop=True)
    demand = flow_data["forecast_p50"].values

    productivity = getattr(config, f"productivity_{flow}")
    initial = getattr(config, f"initial_backlog_{flow}")
    current_staffing = getattr(config, f"current_staffing_{flow}")

    n = len(demand)

    # --- Recommended track ---
    rec_hc, rec_cap = calculate_recommended_capacity(
        demand, initial,
        config.target_backlog_ratio,
        config.backlog_threshold_critical,
        productivity,
        config.hours_per_shift,
    )

    rec_backlog = calculate_daily_backlog(demand, rec_cap, initial)
    rec_dob = calculate_days_of_backlog(rec_backlog["end_backlog"].values, rec_cap)

    # --- Actual track ---
    act_cap = calculate_actual_capacity(n, current_staffing, productivity, config.hours_per_shift)
    act_backlog = calculate_daily_backlog(demand, act_cap, initial)
    act_dob = calculate_days_of_backlog(act_backlog["end_backlog"].values, act_cap)

    result = pd.DataFrame({
        "date": flow_data["date"].values,
        "beg_backlog": rec_backlog["beg_backlog"].values,
        "new_demand": demand,
        "capacity_recommended": rec_cap,
        "capacity_actual": act_cap,
        "end_backlog_recommended": rec_backlog["end_backlog"].values,
        "end_backlog_actual": act_backlog["end_backlog"].values,
        "days_of_backlog_recommended": rec_dob,
        "days_of_backlog_actual": act_dob,
    })

    # Backward-compat aliases (point to recommended track)
    result["capacity"] = result["capacity_recommended"]
    result["end_backlog"] = result["end_backlog_recommended"]
    result["days_of_backlog"] = result["days_of_backlog_recommended"]

    return result


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
