"""Shared test fixtures for hireplanner tests."""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config():
    """Return a valid ClientConfig with typical values."""
    from hireplanner.config.client_config import ClientConfig

    return ClientConfig(
        client_name="Test Warehouse Madrid",
        active_flows=["outbound", "inbound"],
        productivity_inbound=85.0,
        productivity_outbound=120.0,
        hours_per_shift=8,
        overhead_buffer=0.15,
        backlog_threshold_watch=1.0,
        backlog_threshold_critical=2.0,
        initial_backlog_outbound=3500,
        initial_backlog_inbound=1200,
        language="en",
        forecast_horizon=28,
    )


@pytest.fixture
def sample_outbound_only_config():
    """Return a ClientConfig with only outbound flow."""
    from hireplanner.config.client_config import ClientConfig

    return ClientConfig(
        client_name="Test Warehouse Outbound Only",
        active_flows=["outbound"],
        productivity_inbound=85.0,
        productivity_outbound=120.0,
        hours_per_shift=8,
        overhead_buffer=0.15,
        backlog_threshold_watch=1.0,
        backlog_threshold_critical=2.0,
        initial_backlog_outbound=2000,
        initial_backlog_inbound=0,
        language="en",
        forecast_horizon=28,
    )


@pytest.fixture
def sample_forecast_df():
    """Return a 28-day synthetic forecast DataFrame."""
    np.random.seed(42)
    start = date(2026, 3, 1)
    dates = [start + timedelta(days=i) for i in range(28)]
    base_outbound = 5000 + np.random.normal(0, 300, 28)
    base_inbound = 3000 + np.random.normal(0, 200, 28)

    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "date": pd.Timestamp(d),
            "flow": "outbound",
            "forecast_p50": max(0, base_outbound[i]),
            "forecast_p10": max(0, base_outbound[i] * 0.8),
            "forecast_p90": max(0, base_outbound[i] * 1.2),
        })
        rows.append({
            "date": pd.Timestamp(d),
            "flow": "inbound",
            "forecast_p50": max(0, base_inbound[i]),
            "forecast_p10": max(0, base_inbound[i] * 0.8),
            "forecast_p90": max(0, base_inbound[i] * 1.2),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_actuals_df():
    """Return 28-day synthetic actual data for accuracy comparison."""
    np.random.seed(99)
    start = date(2026, 2, 1)
    dates = [start + timedelta(days=i) for i in range(28)]
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in dates],
        "outbound": np.maximum(0, 5000 + np.random.normal(0, 400, 28)),
        "inbound": np.maximum(0, 3000 + np.random.normal(0, 250, 28)),
    })


@pytest.fixture
def sample_history_df():
    """Return 18 months of synthetic daily volume data for ingestion tests."""
    np.random.seed(123)
    start = date(2024, 9, 1)
    n_days = 548  # ~18 months
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Simulate weekly seasonality + trend
    day_indices = np.arange(n_days)
    weekly_pattern = np.sin(2 * np.pi * day_indices / 7) * 500
    trend = day_indices * 2
    outbound = 5000 + weekly_pattern + trend + np.random.normal(0, 300, n_days)
    inbound = 3000 + weekly_pattern * 0.6 + trend * 0.5 + np.random.normal(0, 200, n_days)

    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d in dates],
        "outbound": np.maximum(0, outbound).astype(int),
        "inbound": np.maximum(0, inbound).astype(int),
    })
