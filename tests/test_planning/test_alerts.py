"""Tests for the backlog threshold alert system."""
import numpy as np
import pandas as pd
import pytest

from hireplanner.planning.alerts import (
    classify_backlog_status,
    generate_alert_series,
    summarize_alerts,
)


# ---------------------------------------------------------------------------
# classify_backlog_status
# ---------------------------------------------------------------------------

class TestClassifyBacklogStatus:
    def test_below_watch_is_healthy(self):
        """Value below watch threshold returns Healthy."""
        assert classify_backlog_status(0.5, threshold_watch=1.0, threshold_critical=2.0) == "Healthy"

    def test_between_watch_and_critical_is_watch(self):
        """Value between watch and critical returns Watch."""
        assert classify_backlog_status(1.5, threshold_watch=1.0, threshold_critical=2.0) == "Watch"

    def test_above_critical_is_critical(self):
        """Value above critical threshold returns Critical."""
        assert classify_backlog_status(3.0, threshold_watch=1.0, threshold_critical=2.0) == "Critical"

    def test_exactly_at_watch_is_watch(self):
        """Value exactly at watch threshold returns Watch."""
        assert classify_backlog_status(1.0, threshold_watch=1.0, threshold_critical=2.0) == "Watch"

    def test_exactly_at_critical_is_critical(self):
        """Value exactly at critical threshold returns Critical."""
        assert classify_backlog_status(2.0, threshold_watch=1.0, threshold_critical=2.0) == "Critical"


# ---------------------------------------------------------------------------
# generate_alert_series
# ---------------------------------------------------------------------------

class TestGenerateAlertSeries:
    def test_mixed_scenario(self, sample_config):
        """Alert series correctly classifies a mix of values."""
        # sample_config has watch=1.0, critical=2.0
        days_of_backlog = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        series = generate_alert_series(days_of_backlog, sample_config)

        assert series.iloc[0] == "Healthy"   # 0.0
        assert series.iloc[1] == "Healthy"   # 0.5
        assert series.iloc[2] == "Watch"     # 1.0
        assert series.iloc[3] == "Watch"     # 1.5
        assert series.iloc[4] == "Critical"  # 2.0
        assert series.iloc[5] == "Critical"  # 3.0
        assert series.name == "alert_status"


# ---------------------------------------------------------------------------
# summarize_alerts
# ---------------------------------------------------------------------------

class TestSummarizeAlerts:
    def test_all_healthy(self):
        """When all days are healthy, critical_days=0 and first_critical_date=None."""
        alerts = pd.Series(["Healthy", "Healthy", "Healthy"])
        dob = np.array([0.1, 0.2, 0.3])
        dates = pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"])

        summary = summarize_alerts(alerts, dob, dates)

        assert summary["critical_days"] == 0
        assert summary["watch_days"] == 0
        assert summary["healthy_days"] == 3
        assert summary["first_critical_date"] is None
        assert summary["peak_days_of_backlog"] == pytest.approx(0.3)

    def test_with_critical_days(self):
        """When critical days exist, first_critical_date is populated correctly."""
        alerts = pd.Series(["Healthy", "Watch", "Critical", "Critical"])
        dob = np.array([0.5, 1.5, 2.5, 3.0])
        dates = pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"])

        summary = summarize_alerts(alerts, dob, dates)

        assert summary["critical_days"] == 2
        assert summary["watch_days"] == 1
        assert summary["healthy_days"] == 1
        assert summary["first_critical_date"] == pd.Timestamp("2026-03-03")
        assert summary["peak_days_of_backlog"] == pytest.approx(3.0)
