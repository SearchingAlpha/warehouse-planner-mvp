"""Tests for the LightGBM forecaster."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hireplanner.forecasting.lightgbm_model import (
    LightGBMForecaster,
    _build_features,
    _get_feature_cols,
)


def _make_history(n_days: int = 400, seed: int = 42):
    """Generate synthetic daily history with weekly seasonality."""
    np.random.seed(seed)
    start = date(2024, 1, 1)
    dates = pd.date_range(start, periods=n_days, freq="D")
    day_idx = np.arange(n_days)
    # Weekly seasonality + trend + noise
    values = 5000 + np.sin(2 * np.pi * day_idx / 7) * 500 + day_idx * 2 + np.random.normal(0, 200, n_days)
    return pd.Series(np.maximum(0, values)), dates


class TestBuildFeatures:
    """Tests for feature engineering."""

    def test_lag_columns_created(self):
        history, dates = _make_history(100)
        df = _build_features(history, dates)
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            assert f"lag_{lag}" in df.columns

    def test_rolling_columns_created(self):
        history, dates = _make_history(100)
        df = _build_features(history, dates)
        for window in [7, 14, 28]:
            assert f"rolling_mean_{window}" in df.columns
            assert f"rolling_std_{window}" in df.columns

    def test_calendar_columns_created(self):
        history, dates = _make_history(100)
        df = _build_features(history, dates)
        for col in ["day_of_week", "day_of_month", "week_of_year", "month", "is_weekend"]:
            assert col in df.columns

    def test_feature_cols_excludes_date_and_value(self):
        history, dates = _make_history(100)
        df = _build_features(history, dates)
        cols = _get_feature_cols(df)
        assert "date" not in cols
        assert "value" not in cols
        assert len(cols) > 0

    def test_weekend_values_correct(self):
        history, dates = _make_history(100)
        df = _build_features(history, dates)
        # Check a known Saturday
        sat_mask = pd.to_datetime(df["date"]).dt.dayofweek == 5
        assert (df.loc[sat_mask, "is_weekend"] == 1).all()
        # Monday = 0 → not weekend
        mon_mask = pd.to_datetime(df["date"]).dt.dayofweek == 0
        assert (df.loc[mon_mask, "is_weekend"] == 0).all()


class TestLightGBMForecaster:
    """Tests for the LightGBM forecaster."""

    def test_forecast_returns_correct_keys(self):
        history, dates = _make_history(400)
        model = LightGBMForecaster(n_estimators=50, verbose=-1)
        result = model.forecast(history, dates, horizon=7)
        assert set(result.keys()) == {"p10", "p50", "p90"}

    def test_forecast_correct_length(self):
        history, dates = _make_history(400)
        horizon = 14
        model = LightGBMForecaster(n_estimators=50, verbose=-1)
        result = model.forecast(history, dates, horizon=horizon)
        for key in ("p10", "p50", "p90"):
            assert len(result[key]) == horizon

    def test_forecast_values_non_negative(self):
        history, dates = _make_history(400)
        model = LightGBMForecaster(n_estimators=50, verbose=-1)
        result = model.forecast(history, dates, horizon=7)
        for key in ("p10", "p50", "p90"):
            assert (result[key] >= 0).all(), f"{key} contains negative values"

    def test_quantile_ordering(self):
        """P10 <= P50 <= P90 for most forecast days."""
        history, dates = _make_history(400)
        model = LightGBMForecaster(n_estimators=100, verbose=-1)
        result = model.forecast(history, dates, horizon=14)
        # Allow some tolerance — quantile regression doesn't guarantee strict ordering
        # on every single point, but should hold for the majority
        p10_le_p50 = np.sum(result["p10"] <= result["p50"] + 1)
        p50_le_p90 = np.sum(result["p50"] <= result["p90"] + 1)
        assert p10_le_p50 >= 10, "P10 should be <= P50 for most days"
        assert p50_le_p90 >= 10, "P50 should be <= P90 for most days"

    def test_forecast_with_short_history(self):
        """Model should still work with minimum viable history."""
        history, dates = _make_history(60)
        model = LightGBMForecaster(n_estimators=30, verbose=-1)
        result = model.forecast(history, dates, horizon=7)
        assert len(result["p50"]) == 7

    def test_forecast_accepts_numpy_array(self):
        history, dates = _make_history(200)
        model = LightGBMForecaster(n_estimators=30, verbose=-1)
        result = model.forecast(history.values, dates, horizon=7)
        assert len(result["p50"]) == 7
