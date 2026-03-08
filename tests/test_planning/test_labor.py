import numpy as np
import pandas as pd
import pytest

from hireplanner.config.client_config import ClientConfig
from hireplanner.planning.labor import (
    build_headcount_plan,
    calculate_daily_hours,
    calculate_headcount,
)


class TestCalculateHeadcount:
    def test_known_value(self):
        # 960 units / (120 units/hr * 8 hr) = 1.0, * 1.15 = 1.15 => ceil => 2
        result = calculate_headcount(np.array([960]), productivity=120, hours_per_shift=8, overhead_buffer=0.15)
        assert result[0] == 2

    def test_zero_volume(self):
        result = calculate_headcount(np.array([0, 0]), productivity=100, hours_per_shift=8)
        np.testing.assert_array_equal(result, [0, 0])

    def test_fractional_rounds_up(self):
        # 100 / (100*8) = 0.125, * 1.15 = 0.14375 => ceil => 1
        result = calculate_headcount(np.array([100]), productivity=100, hours_per_shift=8, overhead_buffer=0.15)
        assert result[0] == 1

    def test_exact_integer_stays(self):
        # 800 / (100*8) = 1.0, * 1.0 (no buffer) = 1.0 => ceil => 1
        result = calculate_headcount(np.array([800]), productivity=100, hours_per_shift=8, overhead_buffer=0.0)
        assert result[0] == 1

    def test_multiple_days(self):
        volumes = np.array([960, 480, 0, 1920])
        result = calculate_headcount(volumes, productivity=120, hours_per_shift=8, overhead_buffer=0.15)
        np.testing.assert_array_equal(result, [2, 1, 0, 3])

    def test_accepts_pandas_series(self):
        s = pd.Series([960, 480])
        result = calculate_headcount(s, productivity=120, hours_per_shift=8, overhead_buffer=0.15)
        assert result[0] == 2
        assert result[1] == 1


class TestCalculateDailyHours:
    def test_basic(self):
        hc = np.array([2, 3, 1])
        hours = calculate_daily_hours(hc, hours_per_shift=8)
        np.testing.assert_array_equal(hours, [16, 24, 8])

    def test_zero_headcount(self):
        hc = np.array([0, 0])
        hours = calculate_daily_hours(hc, hours_per_shift=10)
        np.testing.assert_array_equal(hours, [0, 0])


class TestBuildHeadcountPlan:
    @pytest.fixture()
    def config_both_flows(self):
        return ClientConfig(
            client_name="test",
            active_flows=["outbound", "inbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            overhead_buffer=0.15,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=5,
            current_staffing_inbound=3,
        )

    @pytest.fixture()
    def config_outbound_only(self):
        return ClientConfig(
            client_name="test",
            active_flows=["outbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            overhead_buffer=0.15,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=0,
            current_staffing_inbound=0,
        )

    @pytest.fixture()
    def forecast_df(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        ob = pd.DataFrame({"date": dates, "flow": "outbound", "forecast_p50": [960, 480, 1920]})
        ib = pd.DataFrame({"date": dates, "flow": "inbound", "forecast_p50": [800, 400, 1600]})
        return pd.concat([ob, ib], ignore_index=True)

    def test_has_recommended_and_actual_columns(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert "hc_outbound_recommended" in plan.columns
        assert "hc_outbound_actual" in plan.columns
        assert "hc_inbound_recommended" in plan.columns
        assert "hc_inbound_actual" in plan.columns
        assert "hc_total_recommended" in plan.columns
        assert "hc_total_actual" in plan.columns

    def test_backward_compat_aliases(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        pd.testing.assert_series_equal(
            plan["hc_outbound"], plan["hc_outbound_recommended"], check_names=False,
        )
        pd.testing.assert_series_equal(
            plan["hc_total"], plan["hc_total_recommended"], check_names=False,
        )

    def test_actual_staffing_is_constant(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert (plan["hc_outbound_actual"] == 5).all()
        assert (plan["hc_inbound_actual"] == 3).all()
        assert (plan["hc_total_actual"] == 8).all()

    def test_both_flows_length(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert len(plan) == 3

    def test_outbound_only(self, forecast_df, config_outbound_only):
        plan = build_headcount_plan(forecast_df, config_outbound_only)
        assert (plan["hc_inbound_recommended"] == 0).all()
        assert (plan["hc_inbound_actual"] == 0).all()

    def test_recommended_hc_non_negative(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert (plan["hc_outbound_recommended"] >= 0).all()
        assert (plan["hc_inbound_recommended"] >= 0).all()
        assert (plan["hc_total_recommended"] >= 0).all()

    def test_cost_columns_present_when_cost_per_hour_gt_zero(self, forecast_df, config_both_flows):
        config_both_flows.cost_per_hour = 12.50
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert "daily_cost_recommended" in plan.columns
        assert "daily_cost_actual" in plan.columns
        assert "daily_savings" in plan.columns
        assert (plan["daily_cost_recommended"] > 0).any()
        assert (plan["daily_cost_actual"] > 0).all()

    def test_cost_columns_zero_when_cost_per_hour_zero(self, forecast_df, config_both_flows):
        config_both_flows.cost_per_hour = 0.0
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert (plan["daily_cost_recommended"] == 0).all()
        assert (plan["daily_cost_actual"] == 0).all()
        assert (plan["daily_savings"] == 0).all()

    def test_savings_equals_actual_minus_recommended(self, forecast_df, config_both_flows):
        config_both_flows.cost_per_hour = 10.0
        plan = build_headcount_plan(forecast_df, config_both_flows)
        expected = plan["daily_cost_actual"] - plan["daily_cost_recommended"]
        pd.testing.assert_series_equal(plan["daily_savings"], expected, check_names=False)

    def test_weekly_stabilization_default_patterns(self):
        """Default shift patterns (all-week) → each calendar week gets constant HC."""
        # 2026-03-02 is a Monday → 14 days = 2 full weeks
        dates = pd.date_range("2026-03-02", periods=14, freq="D")
        ob = pd.DataFrame({
            "date": dates, "flow": "outbound",
            "forecast_p50": [800, 1200, 600, 1400, 900, 1100, 700,
                             1000, 1300, 500, 1500, 800, 1200, 600],
        })
        ib = pd.DataFrame({
            "date": dates, "flow": "inbound",
            "forecast_p50": [400] * 14,
        })
        forecast_df = pd.concat([ob, ib], ignore_index=True)

        config = ClientConfig(
            client_name="test",
            active_flows=["outbound", "inbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=5,
            current_staffing_inbound=3,
        )
        plan = build_headcount_plan(forecast_df, config)
        # Week 1 (days 0-6) should have constant outbound HC
        week1 = plan["hc_outbound_recommended"].iloc[:7]
        assert len(set(week1)) == 1, f"Week 1 HC not constant: {list(week1)}"
        # Week 2 (days 7-13) should have constant outbound HC
        week2 = plan["hc_outbound_recommended"].iloc[7:]
        assert len(set(week2)) == 1, f"Week 2 HC not constant: {list(week2)}"

    def test_weekday_pattern_zeros_weekends(self):
        """Mon-Fri shift pattern produces zero HC on Saturday and Sunday."""
        # 2026-03-02 is a Monday
        dates = pd.date_range("2026-03-02", periods=7, freq="D")
        ob = pd.DataFrame({
            "date": dates, "flow": "outbound",
            "forecast_p50": [1000] * 7,
        })
        ib = pd.DataFrame({
            "date": dates, "flow": "inbound",
            "forecast_p50": [500] * 7,
        })
        forecast_df = pd.concat([ob, ib], ignore_index=True)

        config = ClientConfig(
            client_name="test",
            active_flows=["outbound", "inbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=5,
            current_staffing_inbound=3,
            shift_patterns=[{"name": "weekday", "days": [0, 1, 2, 3, 4]}],
        )
        plan = build_headcount_plan(forecast_df, config)
        # Saturday (index 5) and Sunday (index 6) should have zero recommended HC
        assert plan["hc_outbound_recommended"].iloc[5] == 0
        assert plan["hc_outbound_recommended"].iloc[6] == 0
        # Weekdays should have HC > 0
        assert (plan["hc_outbound_recommended"].iloc[:5] > 0).all()

    def test_two_rotation_pattern(self):
        """Early+Central pattern creates a realistic daily curve."""
        # 2026-03-02 is a Monday
        dates = pd.date_range("2026-03-02", periods=7, freq="D")
        ob = pd.DataFrame({
            "date": dates, "flow": "outbound",
            "forecast_p50": [1000] * 7,
        })
        ib = pd.DataFrame({
            "date": dates, "flow": "inbound",
            "forecast_p50": [500] * 7,
        })
        forecast_df = pd.concat([ob, ib], ignore_index=True)

        config = ClientConfig(
            client_name="test",
            active_flows=["outbound", "inbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=0,
            current_staffing_inbound=0,
            shift_patterns=[
                {"name": "early", "days": [0, 1, 2, 3]},    # Mon-Thu
                {"name": "central", "days": [2, 3, 4, 5]},   # Wed-Sat
            ],
        )
        plan = build_headcount_plan(forecast_df, config)
        # Sunday (index 6) is uncovered
        assert plan["hc_outbound_recommended"].iloc[6] == 0
        # Weekdays are covered
        assert plan["hc_outbound_recommended"].iloc[0] > 0  # Mon
        assert plan["hc_outbound_recommended"].iloc[4] > 0  # Fri
        # Wed/Thu have overlap → HC ≥ Mon/Tue HC
        assert plan["hc_outbound_recommended"].iloc[2] >= plan["hc_outbound_recommended"].iloc[0]

    def test_per_rotation_actual_staffing(self):
        """Per-rotation actual staffing creates a daily curve, not a flat line."""
        # 2026-03-02 is a Monday
        dates = pd.date_range("2026-03-02", periods=7, freq="D")
        ob = pd.DataFrame({
            "date": dates, "flow": "outbound",
            "forecast_p50": [1000] * 7,
        })
        ib = pd.DataFrame({
            "date": dates, "flow": "inbound",
            "forecast_p50": [500] * 7,
        })
        forecast_df = pd.concat([ob, ib], ignore_index=True)

        config = ClientConfig(
            client_name="test",
            active_flows=["outbound", "inbound"],
            productivity_outbound=120,
            productivity_inbound=100,
            hours_per_shift=8,
            initial_backlog_outbound=0,
            initial_backlog_inbound=0,
            target_backlog_ratio=0.0,
            current_staffing_outbound=[70, 50],
            current_staffing_inbound=[60, 40],
            shift_patterns=[
                {"name": "early", "days": [0, 1, 2, 3]},    # Mon-Thu
                {"name": "central", "days": [2, 3, 4, 5]},   # Wed-Sat
            ],
        )
        plan = build_headcount_plan(forecast_df, config)
        # Outbound actual: Mon=70, Tue=70, Wed=120, Thu=120, Fri=50, Sat=50, Sun=0
        assert plan["hc_outbound_actual"].iloc[0] == 70   # Mon
        assert plan["hc_outbound_actual"].iloc[2] == 120  # Wed (overlap)
        assert plan["hc_outbound_actual"].iloc[4] == 50   # Fri
        assert plan["hc_outbound_actual"].iloc[6] == 0    # Sun
        # Inbound actual: Mon=60, Wed=100, Fri=40, Sun=0
        assert plan["hc_inbound_actual"].iloc[0] == 60
        assert plan["hc_inbound_actual"].iloc[2] == 100
        assert plan["hc_inbound_actual"].iloc[6] == 0
