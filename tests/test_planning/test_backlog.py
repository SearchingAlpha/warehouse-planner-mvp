"""Tests for the backlog calculation engine."""
import math

import numpy as np
import pandas as pd
import pytest

from hireplanner.planning.backlog import (
    apply_shift_patterns,
    calculate_actual_daily_hc,
    calculate_all_backlogs,
    calculate_daily_backlog,
    calculate_daily_capacity,
    calculate_days_of_backlog,
    calculate_flow_backlog,
    calculate_recommended_capacity,
    calculate_actual_capacity,
    solve_rotation_hc,
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
# solve_rotation_hc
# ---------------------------------------------------------------------------

class TestSolveRotationHc:
    def test_single_rotation_all_week(self):
        """Single rotation covering all 7 days: HC = max of daily needs."""
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        needed = np.array([5, 8, 6, 7, 5, 3, 0])  # Mon-Sun
        result = solve_rotation_hc(needed, patterns)
        # The single rotation must cover the peak (8)
        assert result[0] == 8  # Mon
        assert result[5] == 8  # Sat
        assert result[6] == 8  # Sun (covered even though need is 0)

    def test_single_rotation_weekdays_only(self):
        """Mon-Fri rotation: weekends get HC = 0."""
        patterns = [{"name": "weekday", "days": [0, 1, 2, 3, 4]}]
        needed = np.array([5, 8, 6, 7, 5, 3, 2])  # Mon-Sun
        result = solve_rotation_hc(needed, patterns)
        # Weekdays get max of Mon-Fri needs = 8
        assert result[0] == 8
        assert result[4] == 8
        # Weekends are uncovered → 0
        assert result[5] == 0
        assert result[6] == 0

    def test_two_overlapping_rotations_amazon_style(self):
        """Early (Mon-Thu) + Central (Wed-Sat) with overlap on Wed/Thu."""
        patterns = [
            {"name": "early", "days": [0, 1, 2, 3]},    # Mon-Thu
            {"name": "central", "days": [2, 3, 4, 5]},   # Wed-Sat
        ]
        # Need: Mon=5, Tue=5, Wed=10, Thu=10, Fri=5, Sat=5, Sun=0
        needed = np.array([5, 5, 10, 10, 5, 5, 0])
        result = solve_rotation_hc(needed, patterns)

        # Verify all daily needs are met
        for d in range(7):
            if needed[d] > 0:
                assert result[d] >= needed[d], f"Day {d}: {result[d]} < {needed[d]}"
        # Sunday uncovered → 0
        assert result[6] == 0
        # Wed/Thu should have more than Mon/Tue (overlap of both rotations)
        assert result[2] >= result[0]

    def test_zero_demand_produces_zero_hc(self):
        """No demand means no headcount regardless of patterns."""
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        needed = np.zeros(7)
        result = solve_rotation_hc(needed, patterns)
        np.testing.assert_array_equal(result, np.zeros(7))

    def test_demand_on_uncovered_day_gets_zero(self):
        """Demand on a day no rotation covers results in HC = 0 for that day."""
        patterns = [{"name": "weekday", "days": [0, 1, 2, 3, 4]}]
        needed = np.array([5, 5, 5, 5, 5, 10, 10])  # Sat/Sun have high demand
        result = solve_rotation_hc(needed, patterns)
        assert result[5] == 0  # Sat uncovered
        assert result[6] == 0  # Sun uncovered
        # Weekdays still covered
        assert result[0] >= 5


# ---------------------------------------------------------------------------
# apply_shift_patterns
# ---------------------------------------------------------------------------

class TestApplyShiftPatterns:
    def _make_dates(self, start="2026-03-02", n=14):
        """Helper: generate n consecutive dates starting on a Monday."""
        return pd.date_range(start, periods=n, freq="D").values

    def test_all_week_single_rotation_weekly_constant(self):
        """Single all-week rotation → each calendar week gets constant HC."""
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        # 2026-03-02 is a Monday
        dates = self._make_dates("2026-03-02", 14)
        raw_hc = np.array([3, 5, 4, 2, 6, 1, 3,  8, 4, 2, 7, 5, 3, 6])
        new_hc, new_cap = apply_shift_patterns(raw_hc, dates, patterns, 100.0, 8.0)
        # Week 1 peak=6, Week 2 peak=8
        assert len(set(new_hc[:7])) == 1
        assert new_hc[0] == 6
        assert len(set(new_hc[7:])) == 1
        assert new_hc[7] == 8

    def test_weekday_only_gives_zero_on_weekends(self):
        """Mon-Fri rotation produces HC = 0 on Saturday and Sunday."""
        patterns = [{"name": "weekday", "days": [0, 1, 2, 3, 4]}]
        dates = self._make_dates("2026-03-02", 7)  # Mon-Sun
        raw_hc = np.array([5, 5, 5, 5, 5, 5, 5])
        new_hc, _ = apply_shift_patterns(raw_hc, dates, patterns, 100.0, 8.0)
        # Weekdays (Mon-Fri) should have HC
        assert all(new_hc[i] > 0 for i in range(5))
        # Weekend should be 0
        assert new_hc[5] == 0  # Sat
        assert new_hc[6] == 0  # Sun

    def test_two_rotations_overlap(self):
        """Early/Central pattern creates a daily curve, not a flat line."""
        patterns = [
            {"name": "early", "days": [0, 1, 2, 3]},    # Mon-Thu
            {"name": "central", "days": [2, 3, 4, 5]},   # Wed-Sat
        ]
        dates = self._make_dates("2026-03-02", 7)  # Mon-Sun
        # Uniform need
        raw_hc = np.array([10, 10, 10, 10, 10, 10, 0])
        new_hc, _ = apply_shift_patterns(raw_hc, dates, patterns, 100.0, 8.0)
        # Mon/Tue: only early → 10
        # Wed/Thu: early + central overlap → ≥10
        # Fri/Sat: only central → 10
        # Sun: uncovered → 0
        assert new_hc[0] >= 10   # Mon
        assert new_hc[2] >= 10   # Wed (overlap)
        assert new_hc[4] >= 10   # Fri
        assert new_hc[6] == 0    # Sun

    def test_capacity_equals_hc_times_productivity(self):
        """Returned capacity matches HC × productivity × hours."""
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        dates = self._make_dates("2026-03-02", 7)
        raw_hc = np.array([5, 5, 5, 5, 5, 5, 5])
        new_hc, new_cap = apply_shift_patterns(raw_hc, dates, patterns, 100.0, 8.0)
        expected_cap = new_hc.astype(float) * 100.0 * 8.0
        np.testing.assert_array_equal(new_cap, expected_cap)

    def test_partial_week_at_end(self):
        """A partial calendar week at the end is handled correctly."""
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        # 10 days = full week + 3 extra days (Mon-Wed)
        dates = self._make_dates("2026-03-02", 10)
        raw_hc = np.array([3, 5, 4, 2, 6, 1, 3,  9, 4, 2])
        new_hc, _ = apply_shift_patterns(raw_hc, dates, patterns, 100.0, 8.0)
        # Week 1: max=6
        assert new_hc[0] == 6
        # Week 2 (partial): max=9
        assert new_hc[7] == 9
        assert new_hc[9] == 9


# ---------------------------------------------------------------------------
# calculate_actual_daily_hc
# ---------------------------------------------------------------------------

class TestCalculateActualDailyHc:
    def _make_dates(self, start="2026-03-02", n=7):
        return pd.date_range(start, periods=n, freq="D").values

    def test_scalar_staffing_constant(self):
        """Scalar staffing → same HC every day."""
        dates = self._make_dates(n=7)
        patterns = [{"name": "all", "days": [0, 1, 2, 3, 4, 5, 6]}]
        result = calculate_actual_daily_hc(dates, 50, patterns)
        np.testing.assert_array_equal(result, [50] * 7)

    def test_per_rotation_staffing(self):
        """Per-rotation list → daily HC follows shift coverage."""
        dates = self._make_dates("2026-03-02", 7)  # Mon-Sun
        patterns = [
            {"name": "early", "days": [0, 1, 2, 3]},    # Mon-Thu
            {"name": "central", "days": [2, 3, 4, 5]},   # Wed-Sat
        ]
        result = calculate_actual_daily_hc(dates, [70, 50], patterns)
        # Mon=70, Tue=70, Wed=120, Thu=120, Fri=50, Sat=50, Sun=0
        assert result[0] == 70   # Mon (early only)
        assert result[1] == 70   # Tue (early only)
        assert result[2] == 120  # Wed (early + central)
        assert result[3] == 120  # Thu (early + central)
        assert result[4] == 50   # Fri (central only)
        assert result[5] == 50   # Sat (central only)
        assert result[6] == 0    # Sun (uncovered)

    def test_zero_list_staffing(self):
        """All-zero rotation staffing → zero everywhere."""
        dates = self._make_dates(n=7)
        patterns = [
            {"name": "early", "days": [0, 1, 2, 3]},
            {"name": "central", "days": [2, 3, 4, 5]},
        ]
        result = calculate_actual_daily_hc(dates, [0, 0], patterns)
        np.testing.assert_array_equal(result, [0] * 7)


# ---------------------------------------------------------------------------
# calculate_recommended_capacity
# ---------------------------------------------------------------------------

class TestCalculateRecommendedCapacity:
    def test_basic_shape(self):
        """Returns arrays of the correct length."""
        demand = np.array([1000.0, 1200.0, 800.0, 1100.0])
        rec_hc, rec_cap = calculate_recommended_capacity(
            demand, initial_backlog=500.0,
            target_backlog_ratio=0.35, backlog_threshold_critical=2.0,
            productivity=100.0, hours_per_shift=8.0,
        )
        assert len(rec_hc) == 4
        assert len(rec_cap) == 4

    def test_headcount_is_non_negative_integers(self):
        """Recommended headcount should be non-negative integers."""
        demand = np.array([500.0] * 10)
        rec_hc, _ = calculate_recommended_capacity(
            demand, initial_backlog=0.0,
            target_backlog_ratio=0.35, backlog_threshold_critical=2.0,
            productivity=100.0, hours_per_shift=8.0,
        )
        assert all(h >= 0 for h in rec_hc)
        assert all(isinstance(int(h), int) for h in rec_hc)

    def test_capacity_equals_hc_times_productivity(self):
        """Capacity should equal headcount * productivity * hours."""
        demand = np.array([1000.0, 1200.0])
        productivity = 100.0
        hours = 8.0
        rec_hc, rec_cap = calculate_recommended_capacity(
            demand, initial_backlog=0.0,
            target_backlog_ratio=0.35, backlog_threshold_critical=2.0,
            productivity=productivity, hours_per_shift=hours,
        )
        for i in range(len(demand)):
            assert rec_cap[i] == pytest.approx(rec_hc[i] * productivity * hours)

    def test_zero_ratio_clears_backlog(self):
        """With target_backlog_ratio=0, should clear all backlog."""
        demand = np.array([1000.0] * 5)
        rec_hc, rec_cap = calculate_recommended_capacity(
            demand, initial_backlog=5000.0,
            target_backlog_ratio=0.0, backlog_threshold_critical=2.0,
            productivity=100.0, hours_per_shift=8.0,
        )
        # With 0 target, it should staff to clear everything
        assert all(h >= 0 for h in rec_hc)
        # Simulate the backlog - should reach 0
        beg = 5000.0
        for i in range(5):
            beg = max(0, beg + demand[i] - rec_cap[i])
        assert beg == pytest.approx(0.0, abs=1.0)


# ---------------------------------------------------------------------------
# calculate_actual_capacity
# ---------------------------------------------------------------------------

class TestCalculateActualCapacity:
    def test_constant_capacity(self):
        """Actual capacity is constant across all days."""
        cap = calculate_actual_capacity(5, current_staffing=10, productivity=100.0, hours_per_shift=8.0)
        assert len(cap) == 5
        np.testing.assert_array_equal(cap, [8000.0] * 5)

    def test_zero_staffing(self):
        """Zero staffing produces zero capacity."""
        cap = calculate_actual_capacity(3, current_staffing=0, productivity=100.0, hours_per_shift=8.0)
        np.testing.assert_array_equal(cap, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# calculate_flow_backlog (integration with labor module)
# ---------------------------------------------------------------------------

class TestCalculateFlowBacklog:
    def test_produces_correct_columns(self, sample_config, sample_forecast_df):
        """Flow backlog DataFrame has all expected columns."""
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        expected_cols = [
            "date", "beg_backlog", "new_demand",
            "capacity_recommended", "capacity_actual",
            "end_backlog_recommended", "end_backlog_actual",
            "days_of_backlog_recommended", "days_of_backlog_actual",
            "capacity", "end_backlog", "days_of_backlog",
        ]
        assert list(df.columns) == expected_cols

    def test_backward_compat_aliases(self, sample_config, sample_forecast_df):
        """Backward-compat aliases point to recommended track."""
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        pd.testing.assert_series_equal(
            df["capacity"], df["capacity_recommended"], check_names=False,
        )
        pd.testing.assert_series_equal(
            df["end_backlog"], df["end_backlog_recommended"], check_names=False,
        )
        pd.testing.assert_series_equal(
            df["days_of_backlog"], df["days_of_backlog_recommended"], check_names=False,
        )

    def test_length_matches_forecast(self, sample_config, sample_forecast_df):
        """Output length matches number of forecast days for the flow."""
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        n_outbound = len(sample_forecast_df[sample_forecast_df["flow"] == "outbound"])
        assert len(df) == n_outbound

    def test_actual_track_zero_staffing(self, sample_config, sample_forecast_df):
        """With zero current_staffing, actual capacity is zero."""
        assert sample_config.current_staffing_outbound == 0
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        assert (df["capacity_actual"] == 0).all()

    def test_weekday_pattern_zeros_weekends(self, sample_config, sample_forecast_df):
        """Mon-Fri shift pattern produces zero recommended capacity on weekends."""
        sample_config.shift_patterns = [{"name": "weekday", "days": [0, 1, 2, 3, 4]}]
        df = calculate_flow_backlog(sample_config, sample_forecast_df, "outbound")
        weekend_mask = pd.to_datetime(df["date"]).dt.weekday >= 5
        if weekend_mask.any():
            assert (df.loc[weekend_mask, "capacity_recommended"] == 0).all()


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
