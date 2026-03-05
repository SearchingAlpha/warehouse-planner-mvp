"""Tests for the backlog calculation engine."""
import math

import numpy as np
import pandas as pd
import pytest

from hireplanner.planning.backlog import (
    calculate_all_backlogs,
    calculate_daily_backlog,
    calculate_daily_capacity,
    calculate_days_of_backlog,
    calculate_flow_backlog,
)


# ---------------------------------------------------------------------------
# calculate_daily_backlog
# ---------------------------------------------------------------------------

class TestCalculateDailyBacklog:
    def test_hand_calculated_values(self):
        """Verify backlog with known hand-calculated results."""
        demand = np.array([100.0, 80.0, 120.0])
        capacity = np.array([90.0, 90.0, 90.0])
        initial = 50.0

        # Day 0: beg=50, end=max(0, 50-90+100)=60
        # Day 1: beg=60, end=max(0, 60-90+80)=50
        # Day 2: beg=50, end=max(0, 50-90+120)=80
        df = calculate_daily_backlog(demand, capacity, initial)

        assert df["beg_backlog"].iloc[0] == pytest.approx(50.0)
        assert df["end_backlog"].iloc[0] == pytest.approx(60.0)
        assert df["beg_backlog"].iloc[1] == pytest.approx(60.0)
        assert df["end_backlog"].iloc[1] == pytest.approx(50.0)
        assert df["beg_backlog"].iloc[2] == pytest.approx(50.0)
        assert df["end_backlog"].iloc[2] == pytest.approx(80.0)

    def test_backlog_never_negative(self):
        """Backlog should never go below zero even when capacity exceeds demand."""
        demand = np.array([10.0, 10.0, 10.0])
        capacity = np.array([100.0, 100.0, 100.0])
        initial = 0.0

        df = calculate_daily_backlog(demand, capacity, initial)
        assert (df["end_backlog"] >= 0).all()

    def test_multi_day_propagation(self):
        """Each day's end_backlog equals next day's beg_backlog."""
        demand = np.array([200.0, 150.0, 300.0, 100.0])
        capacity = np.array([180.0, 180.0, 180.0, 180.0])
        initial = 100.0

        df = calculate_daily_backlog(demand, capacity, initial)
        for i in range(len(demand) - 1):
            assert df["end_backlog"].iloc[i] == pytest.approx(df["beg_backlog"].iloc[i + 1])

    def test_zero_initial_backlog(self):
        """Zero initial backlog: first day beg_backlog is 0."""
        demand = np.array([50.0, 60.0])
        capacity = np.array([40.0, 40.0])

        df = calculate_daily_backlog(demand, capacity, 0.0)
        assert df["beg_backlog"].iloc[0] == 0.0
        # Day 0 end: max(0, 0-40+50) = 10
        assert df["end_backlog"].iloc[0] == pytest.approx(10.0)

    def test_high_initial_backlog_decreases(self):
        """When capacity > demand, a high initial backlog should decrease."""
        demand = np.array([50.0, 50.0, 50.0, 50.0])
        capacity = np.array([200.0, 200.0, 200.0, 200.0])
        initial = 500.0

        df = calculate_daily_backlog(demand, capacity, initial)
        # Day 0: end = max(0, 500-200+50) = 350
        # Day 1: end = max(0, 350-200+50) = 200
        # Day 2: end = max(0, 200-200+50) = 50
        # Day 3: end = max(0, 50-200+50)  = 0
        assert df["end_backlog"].iloc[0] == pytest.approx(350.0)
        assert df["end_backlog"].iloc[1] == pytest.approx(200.0)
        assert df["end_backlog"].iloc[2] == pytest.approx(50.0)
        assert df["end_backlog"].iloc[3] == pytest.approx(0.0)

    def test_zero_demand_keeps_backlog_decreasing(self):
        """With zero demand, backlog should decrease by capacity each day."""
        demand = np.array([0.0, 0.0, 0.0])
        capacity = np.array([100.0, 100.0, 100.0])
        initial = 250.0

        df = calculate_daily_backlog(demand, capacity, initial)
        assert df["end_backlog"].iloc[0] == pytest.approx(150.0)
        assert df["end_backlog"].iloc[1] == pytest.approx(50.0)
        assert df["end_backlog"].iloc[2] == pytest.approx(0.0)  # clamped at 0

    def test_zero_capacity_causes_backlog_growth(self):
        """With zero capacity, backlog grows by demand each day."""
        demand = np.array([100.0, 200.0, 300.0])
        capacity = np.array([0.0, 0.0, 0.0])
        initial = 50.0

        df = calculate_daily_backlog(demand, capacity, initial)
        assert df["end_backlog"].iloc[0] == pytest.approx(150.0)
        assert df["end_backlog"].iloc[1] == pytest.approx(350.0)
        assert df["end_backlog"].iloc[2] == pytest.approx(650.0)

    def test_returns_correct_columns(self):
        """Returned DataFrame has the expected columns."""
        df = calculate_daily_backlog(np.array([10.0]), np.array([10.0]), 0.0)
        assert list(df.columns) == ["beg_backlog", "new_demand", "capacity", "end_backlog"]


# ---------------------------------------------------------------------------
# calculate_daily_capacity
# ---------------------------------------------------------------------------

class TestCalculateDailyCapacity:
    def test_basic_calculation(self):
        """capacity = headcount * productivity * hours_per_shift."""
        hc = np.array([5, 10, 3])
        productivity = 120.0
        hours = 8.0

        cap = calculate_daily_capacity(hc, productivity, hours)
        expected = np.array([5 * 120 * 8, 10 * 120 * 8, 3 * 120 * 8], dtype=float)
        np.testing.assert_array_almost_equal(cap, expected)


# ---------------------------------------------------------------------------
# calculate_days_of_backlog
# ---------------------------------------------------------------------------

class TestCalculateDaysOfBacklog:
    def test_known_values(self):
        """Days of backlog with uniform capacity over 7-day window."""
        end_backlog = np.array([1000.0, 500.0, 0.0])
        capacity = np.array([500.0, 500.0, 500.0])

        result = calculate_days_of_backlog(end_backlog, capacity)
        # Day 0: 1000 / mean(cap[1:3]) = 1000/500 = 2.0
        assert result[0] == pytest.approx(2.0)
        # Day 1: 500 / mean(cap[2:3]) = 500/500 = 1.0
        assert result[1] == pytest.approx(1.0)
        # Day 2 (last): 0 / cap[2] = 0.0
        assert result[2] == pytest.approx(0.0)

    def test_end_of_horizon_fewer_than_window(self):
        """At end of horizon with < 7 days remaining, uses available days."""
        end_backlog = np.array([0.0] * 5 + [700.0])
        capacity = np.array([100.0] * 6)

        result = calculate_days_of_backlog(end_backlog, capacity)
        # Last day (index 5): start=6 >= n=6, so uses cap[5]=100
        assert result[5] == pytest.approx(700.0 / 100.0)


# ---------------------------------------------------------------------------
# calculate_flow_backlog (integration with labor module)
# ---------------------------------------------------------------------------

class TestCalculateFlowBacklog:
    def test_produces_correct_columns(self, sample_config, sample_forecast_df):
        """Flow backlog DataFrame has all expected columns."""
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        expected_cols = ["date", "beg_backlog", "new_demand", "capacity", "end_backlog", "days_of_backlog"]
        assert list(df.columns) == expected_cols

    def test_length_matches_forecast(self, sample_config, sample_forecast_df):
        """Output length matches number of forecast days for the flow."""
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        n_outbound = len(sample_forecast_df[sample_forecast_df["flow"] == "outbound"])
        assert len(df) == n_outbound


# ---------------------------------------------------------------------------
# calculate_all_backlogs
# ---------------------------------------------------------------------------

class TestCalculateAllBacklogs:
    def test_both_flows(self, sample_config, sample_forecast_df):
        """With both flows active, returns dict with both keys."""
        result = calculate_all_backlogs(sample_config, sample_forecast_df)
        assert "outbound" in result
        assert "inbound" in result
        assert len(result) == 2

    def test_single_flow(self, sample_outbound_only_config, sample_forecast_df):
        """With single flow active, returns dict with one key."""
        result = calculate_all_backlogs(sample_outbound_only_config, sample_forecast_df)
        assert "outbound" in result
        assert "inbound" not in result
        assert len(result) == 1
