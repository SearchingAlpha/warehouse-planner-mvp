"""Backlog threshold alerts for warehouse planning."""
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd


def classify_backlog_status(
    days_of_backlog: float,
    threshold_watch: float,
    threshold_critical: float,
) -> str:
    """Classify a single days-of-backlog value into alert status.

    Returns: "Healthy", "Watch", or "Critical"
    """
    if days_of_backlog >= threshold_critical:
        return "Critical"
    elif days_of_backlog >= threshold_watch:
        return "Watch"
    return "Healthy"


def generate_alert_series(
    days_of_backlog: np.ndarray | pd.Series,
    config,  # ClientConfig
) -> pd.Series:
    """Apply threshold classification to every day.

    Returns Series of status strings.
    """
    values = np.asarray(days_of_backlog, dtype=float)
    statuses = [
        classify_backlog_status(v, config.backlog_threshold_watch, config.backlog_threshold_critical)
        for v in values
    ]
    return pd.Series(statuses, name="alert_status")


def summarize_alerts(
    alert_series: pd.Series,
    days_of_backlog: np.ndarray | pd.Series,
    dates: Optional[np.ndarray | pd.Series] = None,
) -> dict:
    """Summarize alert information for the executive summary.

    Returns dict with:
        critical_days: int
        watch_days: int
        healthy_days: int
        first_critical_date: date or None
        peak_days_of_backlog: float
    """
    backlog_vals = np.asarray(days_of_backlog, dtype=float)

    critical_mask = alert_series == "Critical"
    watch_mask = alert_series == "Watch"
    healthy_mask = alert_series == "Healthy"

    first_critical = None
    if critical_mask.any() and dates is not None:
        dates_arr = pd.to_datetime(dates)
        idx = critical_mask.idxmax() if critical_mask.any() else None
        if idx is not None:
            first_critical = dates_arr.iloc[idx] if hasattr(dates_arr, 'iloc') else dates_arr[idx]

    return {
        "critical_days": int(critical_mask.sum()),
        "watch_days": int(watch_mask.sum()),
        "healthy_days": int(healthy_mask.sum()),
        "first_critical_date": first_critical,
        "peak_days_of_backlog": float(backlog_vals.max()) if len(backlog_vals) > 0 else 0.0,
    }
