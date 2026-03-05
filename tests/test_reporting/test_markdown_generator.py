"""Tests for the Markdown + PNG report generator."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hireplanner.config.client_config import ClientConfig
from hireplanner.reporting.markdown_generator import (
    generate_markdown_report,
    _df_to_markdown,
    _alert_indicator,
)


# ---------------------------------------------------------------------------
# Locale YAML content (embedded so tests are self-contained)
# ---------------------------------------------------------------------------

EN_YAML = """\
tabs:
  executive_summary: "Executive Summary"
  daily_forecast: "Daily Forecast"
  backlog_projection: "Backlog Projection"
  headcount_plan: "Headcount Plan"
  accuracy_report: "Accuracy Report"

headers:
  date: "Date"
  day_of_week: "Day"
  forecast_p50: "Forecast"
  forecast_p10: "Lower (P10)"
  forecast_p90: "Upper (P90)"
  beg_backlog: "Beg Backlog"
  new_demand: "New Demand"
  capacity: "Capacity"
  capacity_recommended: "Capacity (Rec)"
  capacity_actual: "Capacity (Actual)"
  end_backlog: "End Backlog"
  end_backlog_recommended: "End Backlog (Rec)"
  end_backlog_actual: "End Backlog (Actual)"
  days_of_backlog: "Days of Backlog"
  days_of_backlog_recommended: "Days Backlog (Rec)"
  days_of_backlog_actual: "Days Backlog (Actual)"
  alert_status: "Status"
  hc_inbound: "HC Inbound"
  hc_outbound: "HC Outbound"
  hc_total: "HC Total"
  hc_recommended: "(Rec)"
  hc_actual: "(Actual)"
  target_backlog: "Target Backlog"
  forecast_col: "Forecast"
  actual: "Actual"
  error: "Abs Error"
  pct_error: "% Error"

alerts:
  healthy: "Healthy"
  watch: "Watch"
  critical: "Critical"

labels:
  client: "Client"
  report_date: "Report Date"
  forecast_period: "Forecast Period"
  wape: "WAPE"
  mape: "MAPE"
  mae: "MAE"
  avg_days_of_backlog: "Avg Days of Backlog"
  peak_headcount: "Peak Headcount"
  critical_alert_days: "Critical Alert Days"
  trend: "Week-over-Week Trend"
  improving: "Improving"
  stable: "Stable"
  degrading: "Degrading"
  no_accuracy_data: "No historical accuracy data available yet."
  outbound: "Outbound"
  inbound: "Inbound"
  total: "Total"
  weekly_summary: "Weekly Summary"
  avg_daily_hc: "Avg Daily HC"
  total_hours: "Total Hours"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_locales_dir(tmp_path):
    locales_dir = tmp_path / "locales"
    locales_dir.mkdir(exist_ok=True)
    (locales_dir / "en.yaml").write_text(EN_YAML, encoding="utf-8")
    return locales_dir


def _make_config(active_flows=None, current_staffing_outbound=0, current_staffing_inbound=0):
    if active_flows is None:
        active_flows = ["outbound", "inbound"]
    return ClientConfig(
        client_name="Test Warehouse",
        active_flows=active_flows,
        productivity_inbound=85.0,
        productivity_outbound=120.0,
        hours_per_shift=8,
        overhead_buffer=0.15,
        backlog_threshold_watch=1.0,
        backlog_threshold_critical=2.0,
        initial_backlog_outbound=1000,
        initial_backlog_inbound=500,
        target_backlog_ratio=0.35,
        current_staffing_outbound=current_staffing_outbound,
        current_staffing_inbound=current_staffing_inbound,
        language="en",
        forecast_horizon=28,
    )


def _make_forecast_df(n_days=7, flows=("outbound", "inbound")):
    start = date(2026, 3, 1)
    rows = []
    for i in range(n_days):
        d = pd.Timestamp(start + timedelta(days=i))
        for flow in flows:
            base = 5000.0 if flow == "outbound" else 3000.0
            rows.append({
                "date": d,
                "flow": flow,
                "forecast_p50": base + i * 10,
                "forecast_p10": base * 0.8 + i * 8,
                "forecast_p90": base * 1.2 + i * 12,
            })
    return pd.DataFrame(rows)


def _make_backlog_df(n_days=7):
    start = date(2026, 3, 1)
    rows = []
    for i in range(n_days):
        d = pd.Timestamp(start + timedelta(days=i))
        rows.append({
            "date": d,
            "beg_backlog": 1000.0 + i * 50,
            "new_demand": 5000.0,
            "capacity_recommended": 4800.0,
            "capacity_actual": 4000.0,
            "end_backlog_recommended": 1200.0 + i * 50,
            "end_backlog_actual": 2000.0 + i * 100,
            "days_of_backlog_recommended": 0.25 + i * 0.01,
            "days_of_backlog_actual": 0.42 + i * 0.02,
            "capacity": 4800.0,
            "end_backlog": 1200.0 + i * 50,
            "days_of_backlog": 0.25 + i * 0.01,
        })
    return pd.DataFrame(rows)


def _make_headcount_df(n_days=7):
    start = date(2026, 3, 1)
    rows = []
    for i in range(n_days):
        d = pd.Timestamp(start + timedelta(days=i))
        rows.append({
            "date": d,
            "hc_outbound_recommended": 10 + i,
            "hc_outbound_actual": 8,
            "hc_inbound_recommended": 5 + i,
            "hc_inbound_actual": 4,
            "hc_total_recommended": 15 + 2 * i,
            "hc_total_actual": 12,
            "hc_outbound": 10 + i,
            "hc_inbound": 5 + i,
            "hc_total": 15 + 2 * i,
        })
    return pd.DataFrame(rows)


def _make_accuracy_data(n_days=7):
    start = date(2026, 2, 1)
    rows = []
    for i in range(n_days):
        d = pd.Timestamp(start + timedelta(days=i))
        forecast_val = 5000.0 + i * 10
        actual_val = 5100.0 + i * 5
        rows.append({
            "date": d,
            "forecast": forecast_val,
            "actual": actual_val,
            "absolute_error": abs(forecast_val - actual_val),
            "percentage_error": abs(forecast_val - actual_val) / actual_val,
        })
    return {
        "comparison": pd.DataFrame(rows),
        "metrics": {"wape": 0.02, "mape": 0.021, "mae": 100.0},
        "trend": "stable",
    }


def _generate_report(tmp_path, config=None, forecast_df=None, backlog_dfs=None,
                      headcount_df=None, accuracy_data=None, alert_summary=None,
                      alert_series_dict=None):
    if config is None:
        config = _make_config()
    if forecast_df is None:
        forecast_df = _make_forecast_df()
    if backlog_dfs is None:
        backlog_dfs = {
            "outbound": _make_backlog_df(),
            "inbound": _make_backlog_df(),
        }
    if headcount_df is None:
        headcount_df = _make_headcount_df()

    locales_dir = _make_locales_dir(tmp_path)
    output_dir = tmp_path / "output" / "report_dir"

    result_path = generate_markdown_report(
        config=config,
        forecast_df=forecast_df,
        backlog_dfs=backlog_dfs,
        headcount_df=headcount_df,
        accuracy_data=accuracy_data,
        alert_summary=alert_summary,
        alert_series_dict=alert_series_dict,
        output_dir=output_dir,
        locales_dir=locales_dir,
    )
    return result_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_df_to_markdown(self):
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        md = _df_to_markdown(df)
        assert "| A | B |" in md
        assert "| --- | --- |" in md
        assert "| 1 | x |" in md

    def test_alert_indicator_critical(self):
        assert _alert_indicator("Critical") == "[CRITICAL]"

    def test_alert_indicator_watch(self):
        assert _alert_indicator("Watch") == "[!!]"

    def test_alert_indicator_healthy(self):
        assert _alert_indicator("Healthy") == "[OK]"


class TestMarkdownGenerator:
    def test_generates_report_md(self, tmp_path):
        result_path = _generate_report(tmp_path)
        assert Path(result_path).exists()
        assert result_path.endswith("report.md")

    def test_generates_figures_directory(self, tmp_path):
        result_path = _generate_report(tmp_path)
        figures_dir = Path(result_path).parent / "figures"
        assert figures_dir.exists()
        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) > 0

    def test_report_has_five_sections(self, tmp_path):
        result_path = _generate_report(tmp_path)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "Executive Summary" in content
        assert "Daily Forecast" in content
        assert "Backlog Projection" in content
        assert "Headcount Plan" in content
        assert "Accuracy Report" in content

    def test_report_has_client_name(self, tmp_path):
        result_path = _generate_report(tmp_path)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "Test Warehouse" in content

    def test_report_references_png_images(self, tmp_path):
        result_path = _generate_report(tmp_path)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "![" in content
        assert "figures/" in content
        assert ".png" in content

    def test_accuracy_no_data_message(self, tmp_path):
        result_path = _generate_report(tmp_path, accuracy_data=None)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "No historical accuracy data" in content

    def test_accuracy_with_data(self, tmp_path):
        result_path = _generate_report(tmp_path, accuracy_data=_make_accuracy_data())
        content = Path(result_path).read_text(encoding="utf-8")
        assert "WAPE" in content
        assert "2.0%" in content

    def test_dual_capacity_columns_with_staffing(self, tmp_path):
        config = _make_config(current_staffing_outbound=10, current_staffing_inbound=5)
        result_path = _generate_report(tmp_path, config=config)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "(Rec)" in content
        assert "(Actual)" in content

    def test_no_actual_columns_when_zero_staffing(self, tmp_path):
        config = _make_config(current_staffing_outbound=0, current_staffing_inbound=0)
        result_path = _generate_report(tmp_path, config=config)
        content = Path(result_path).read_text(encoding="utf-8")
        # Backlog section should not have actual columns
        assert "End Backlog (Actual)" not in content

    def test_alert_indicators_in_backlog(self, tmp_path):
        alert_series_dict = {
            "outbound": pd.Series(["Healthy", "Watch", "Critical"] + ["Healthy"] * 4),
            "inbound": pd.Series(["Healthy"] * 7),
        }
        result_path = _generate_report(tmp_path, alert_series_dict=alert_series_dict)
        content = Path(result_path).read_text(encoding="utf-8")
        assert "[OK]" in content
        assert "[!!]" in content
        assert "[CRITICAL]" in content
