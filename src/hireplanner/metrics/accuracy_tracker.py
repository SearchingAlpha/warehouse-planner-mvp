"""Forecast vs actual accuracy tracking with historical logging."""
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from hireplanner.metrics.evaluation import wape, mape, mae


def compare_forecast_to_actual(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    flow: str,
) -> pd.DataFrame:
    """Compare previous forecast against actual volumes.

    Args:
        forecast_df: DataFrame with columns date, flow, forecast_p50
        actual_df: DataFrame with columns date, and flow name as column (e.g. "outbound")
        flow: Which flow to compare

    Returns:
        DataFrame with columns: date, forecast, actual, absolute_error, percentage_error
    """
    fc = forecast_df[forecast_df["flow"] == flow][["date", "forecast_p50"]].copy()
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc.rename(columns={"forecast_p50": "forecast"})

    act = actual_df[["date", flow]].copy()
    act["date"] = pd.to_datetime(act["date"])
    act = act.rename(columns={flow: "actual"})

    merged = fc.merge(act, on="date", how="inner")
    merged["absolute_error"] = np.abs(merged["forecast"] - merged["actual"])
    merged["percentage_error"] = np.where(
        merged["actual"] != 0,
        merged["absolute_error"] / merged["actual"],
        0.0,
    )
    return merged.reset_index(drop=True)


def calculate_accuracy_metrics(comparison_df: pd.DataFrame) -> dict:
    """Calculate accuracy metrics from a comparison DataFrame.

    Returns dict with wape, mape, mae, period_start, period_end.
    """
    actual = comparison_df["actual"].values
    forecast = comparison_df["forecast"].values

    return {
        "wape": wape(actual, forecast),
        "mape": mape(actual, forecast),
        "mae": mae(actual, forecast),
        "period_start": comparison_df["date"].min(),
        "period_end": comparison_df["date"].max(),
    }


def append_accuracy_log(
    client_name: str,
    metrics: dict,
    log_dir: str | Path,
    run_date: Optional[date] = None,
) -> Path:
    """Append accuracy metrics to the client's historical accuracy log CSV.

    Creates the file if it doesn't exist.
    Returns path to the log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    safe_name = client_name.lower().replace(" ", "_")
    log_path = log_dir / f"{safe_name}_accuracy.csv"

    if run_date is None:
        run_date = date.today()

    row = pd.DataFrame([{
        "run_date": run_date,
        "period_start": metrics.get("period_start"),
        "period_end": metrics.get("period_end"),
        "wape": metrics.get("wape"),
        "mape": metrics.get("mape"),
        "mae": metrics.get("mae"),
    }])

    if log_path.exists():
        existing = pd.read_csv(log_path)
        updated = pd.concat([existing, row], ignore_index=True)
    else:
        updated = row

    updated.to_csv(log_path, index=False)
    return log_path


def load_accuracy_log(
    client_name: str,
    log_dir: str | Path,
) -> Optional[pd.DataFrame]:
    """Load the historical accuracy log for a client. Returns None if no log exists."""
    log_dir = Path(log_dir)
    safe_name = client_name.lower().replace(" ", "_")
    log_path = log_dir / f"{safe_name}_accuracy.csv"

    if not log_path.exists():
        return None
    return pd.read_csv(log_path, parse_dates=["run_date", "period_start", "period_end"])


def check_accuracy_degradation(current_wape: float, threshold: float = 0.15) -> bool:
    """Return True if accuracy has degraded past threshold."""
    return current_wape > threshold


def get_accuracy_trend(
    log_df: pd.DataFrame,
    last_n: int = 4,
) -> str:
    """Determine accuracy trend from historical log.

    Returns: "improving", "stable", or "degrading"
    """
    if log_df is None or len(log_df) < 2:
        return "stable"

    recent = log_df.tail(last_n)["wape"].values
    if len(recent) < 2:
        return "stable"

    # Simple linear trend
    diffs = np.diff(recent)
    avg_diff = np.mean(diffs)

    if avg_diff < -0.01:  # WAPE decreasing by >1pp on average
        return "improving"
    elif avg_diff > 0.01:  # WAPE increasing by >1pp on average
        return "degrading"
    return "stable"


def save_forecast(
    client_name: str,
    forecast_df: pd.DataFrame,
    run_date: date,
    log_dir: str | Path,
) -> Path:
    """Save forecast for future accuracy comparison."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_name = client_name.lower().replace(" ", "_")
    path = log_dir / f"{safe_name}_forecast_{run_date.isoformat()}.csv"
    forecast_df.to_csv(path, index=False)
    return path


def load_previous_forecast(
    client_name: str,
    log_dir: str | Path,
) -> Optional[pd.DataFrame]:
    """Load the most recent saved forecast for a client."""
    log_dir = Path(log_dir)
    safe_name = client_name.lower().replace(" ", "_")

    pattern = f"{safe_name}_forecast_*.csv"
    files = sorted(log_dir.glob(pattern))

    if not files:
        return None

    return pd.read_csv(files[-1], parse_dates=["date"])
