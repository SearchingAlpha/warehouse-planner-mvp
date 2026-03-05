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
        # Need volume such that raw is exactly integer: unlikely, but large volume test
        # 800 / (100*8) = 1.0, * 1.0 (no buffer) = 1.0 => ceil => 1
        result = calculate_headcount(np.array([800]), productivity=100, hours_per_shift=8, overhead_buffer=0.0)
        assert result[0] == 1

    def test_multiple_days(self):
        volumes = np.array([960, 480, 0, 1920])
        result = calculate_headcount(volumes, productivity=120, hours_per_shift=8, overhead_buffer=0.15)
        # 960/(120*8)*1.15 = 1.15 => 2
        # 480/(120*8)*1.15 = 0.575 => 1
        # 0 => 0
        # 1920/(120*8)*1.15 = 2.3 => 3
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
        )

    @pytest.fixture()
    def forecast_df(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        ob = pd.DataFrame({"date": dates, "flow": "outbound", "forecast_p50": [960, 480, 1920]})
        ib = pd.DataFrame({"date": dates, "flow": "inbound", "forecast_p50": [800, 400, 1600]})
        return pd.concat([ob, ib], ignore_index=True)

    def test_both_flows(self, forecast_df, config_both_flows):
        plan = build_headcount_plan(forecast_df, config_both_flows)
        assert list(plan.columns) == ["date", "hc_outbound", "hc_inbound", "hc_total"]
        assert len(plan) == 3
        # outbound: 960/(120*8)*1.15=1.15=>2, 480=>1, 1920=>3
        np.testing.assert_array_equal(plan["hc_outbound"].values, [2, 1, 3])
        # inbound: 800/(100*8)*1.15=1.15=>2, 400=>1, 1600=>3
        np.testing.assert_array_equal(plan["hc_inbound"].values, [2, 1, 3])
        np.testing.assert_array_equal(plan["hc_total"].values, [4, 2, 6])

    def test_outbound_only(self, forecast_df, config_outbound_only):
        plan = build_headcount_plan(forecast_df, config_outbound_only)
        np.testing.assert_array_equal(plan["hc_outbound"].values, [2, 1, 3])
        np.testing.assert_array_equal(plan["hc_inbound"].values, [0, 0, 0])
        np.testing.assert_array_equal(plan["hc_total"].values, [2, 1, 3])
