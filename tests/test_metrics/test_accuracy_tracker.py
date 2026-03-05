"""Tests for the accuracy tracking module."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hireplanner.metrics.accuracy_tracker import (
    append_accuracy_log,
    calculate_accuracy_metrics,
    check_accuracy_degradation,
    compare_forecast_to_actual,
    get_accuracy_trend,
    load_accuracy_log,
    load_previous_forecast,
    save_forecast,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def forecast_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "flow": ["outbound", "outbound", "outbound"],
        "forecast_p50": [100.0, 200.0, 150.0],
    })


@pytest.fixture
def actual_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "outbound": [110.0, 190.0, 160.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompareForecaseToActual:
    def test_joins_correctly_on_date(self, forecast_df, actual_df):
        result = compare_forecast_to_actual(forecast_df, actual_df, "outbound")
        assert len(result) == 3
        assert list(result.columns) == [
            "date", "forecast", "actual", "absolute_error", "percentage_error",
        ]
        # First row: forecast=100, actual=110
        assert result.loc[0, "forecast"] == 100.0
        assert result.loc[0, "actual"] == 110.0
        assert np.isclose(result.loc[0, "absolute_error"], 10.0)

    def test_handles_non_overlapping_dates(self, forecast_df):
        actual_different = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-04", "2025-01-05"]),
            "outbound": [300.0, 400.0],
        })
        result = compare_forecast_to_actual(forecast_df, actual_different, "outbound")
        assert len(result) == 0


class TestCalculateAccuracyMetrics:
    def test_returns_correct_keys(self, forecast_df, actual_df):
        comparison = compare_forecast_to_actual(forecast_df, actual_df, "outbound")
        metrics = calculate_accuracy_metrics(comparison)
        assert "wape" in metrics
        assert "mape" in metrics
        assert "mae" in metrics
        assert "period_start" in metrics
        assert "period_end" in metrics


class TestAppendAccuracyLog:
    def test_creates_new_file(self, tmp_path):
        metrics = {
            "wape": 0.05,
            "mape": 0.06,
            "mae": 10.0,
            "period_start": pd.Timestamp("2025-01-01"),
            "period_end": pd.Timestamp("2025-01-07"),
        }
        log_path = append_accuracy_log("Test Client", metrics, tmp_path, run_date=date(2025, 1, 8))
        assert log_path.exists()
        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert df.loc[0, "wape"] == 0.05

    def test_appends_to_existing_file(self, tmp_path):
        metrics_1 = {
            "wape": 0.05, "mape": 0.06, "mae": 10.0,
            "period_start": pd.Timestamp("2025-01-01"),
            "period_end": pd.Timestamp("2025-01-07"),
        }
        metrics_2 = {
            "wape": 0.08, "mape": 0.09, "mae": 15.0,
            "period_start": pd.Timestamp("2025-01-08"),
            "period_end": pd.Timestamp("2025-01-14"),
        }
        append_accuracy_log("Test Client", metrics_1, tmp_path, run_date=date(2025, 1, 8))
        append_accuracy_log("Test Client", metrics_2, tmp_path, run_date=date(2025, 1, 15))
        log_path = tmp_path / "test_client_accuracy.csv"
        df = pd.read_csv(log_path)
        assert len(df) == 2


class TestLoadAccuracyLog:
    def test_returns_none_when_no_file(self, tmp_path):
        result = load_accuracy_log("nonexistent", tmp_path)
        assert result is None

    def test_reads_existing_log(self, tmp_path):
        metrics = {
            "wape": 0.05, "mape": 0.06, "mae": 10.0,
            "period_start": pd.Timestamp("2025-01-01"),
            "period_end": pd.Timestamp("2025-01-07"),
        }
        append_accuracy_log("My Client", metrics, tmp_path, run_date=date(2025, 1, 8))
        result = load_accuracy_log("My Client", tmp_path)
        assert result is not None
        assert len(result) == 1
        assert result.loc[0, "wape"] == 0.05


class TestCheckAccuracyDegradation:
    def test_detects_degradation(self):
        assert check_accuracy_degradation(0.20, threshold=0.15) is True
        assert check_accuracy_degradation(0.10, threshold=0.15) is False
        assert check_accuracy_degradation(0.15, threshold=0.15) is False


class TestGetAccuracyTrend:
    def test_returns_improving_stable_degrading(self):
        # Improving: WAPE going down
        improving_df = pd.DataFrame({"wape": [0.20, 0.15, 0.10, 0.05]})
        assert get_accuracy_trend(improving_df) == "improving"

        # Degrading: WAPE going up
        degrading_df = pd.DataFrame({"wape": [0.05, 0.10, 0.15, 0.20]})
        assert get_accuracy_trend(degrading_df) == "degrading"

        # Stable: WAPE roughly flat
        stable_df = pd.DataFrame({"wape": [0.10, 0.10, 0.105, 0.10]})
        assert get_accuracy_trend(stable_df) == "stable"

        # None input
        assert get_accuracy_trend(None) == "stable"


class TestSaveAndLoadForecast:
    def test_roundtrip(self, tmp_path):
        forecast = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "flow": ["outbound", "outbound"],
            "forecast_p50": [100.0, 200.0],
        })
        save_forecast("Acme Corp", forecast, date(2025, 1, 1), tmp_path)
        loaded = load_previous_forecast("Acme Corp", tmp_path)
        assert loaded is not None
        assert len(loaded) == 2
        assert list(loaded.columns) == ["date", "flow", "forecast_p50"]
