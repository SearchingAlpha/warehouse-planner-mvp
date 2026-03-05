"""Tests for the pipeline runner."""
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_config_file(tmp_path):
    """Create a minimal client config YAML."""
    config_data = {
        "client_name": "Test Pipeline Client",
        "active_flows": ["outbound"],
        "productivity_inbound": 85.0,
        "productivity_outbound": 120.0,
        "hours_per_shift": 8,
        "overhead_buffer": 0.15,
        "backlog_threshold_watch": 1.0,
        "backlog_threshold_critical": 2.0,
        "initial_backlog_outbound": 1000,
        "initial_backlog_inbound": 0,
        "target_backlog_ratio": 0.35,
        "current_staffing_outbound": 0,
        "current_staffing_inbound": 0,
        "language": "en",
        "forecast_horizon": 7,  # Short horizon for fast tests
        "cost_per_hour": 0.0,
    }
    config_path = tmp_path / "test_client.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return str(config_path)


def _make_data_file(tmp_path):
    """Create a minimal volume data CSV (400 days)."""
    start = date(2025, 1, 1)
    dates = [start + timedelta(days=i) for i in range(400)]
    np.random.seed(42)
    df = pd.DataFrame({
        "date": dates,
        "outbound": np.maximum(0, 5000 + np.random.normal(0, 300, 400)).astype(int),
        "inbound": np.maximum(0, 3000 + np.random.normal(0, 200, 400)).astype(int),
    })
    data_path = tmp_path / "volumes.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)


def _make_locale_files(tmp_path):
    """Create minimal English locale file."""
    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()

    en_locale = {
        "tabs": {
            "executive_summary": "Executive Summary",
            "daily_forecast": "Daily Forecast",
            "backlog_projection": "Backlog Projection",
            "headcount_plan": "Headcount Plan",
            "accuracy_report": "Accuracy Report",
        },
        "headers": {
            "date": "Date",
            "day_of_week": "Day",
            "forecast_p50": "Forecast",
            "forecast_p10": "Lower (P10)",
            "forecast_p90": "Upper (P90)",
            "beg_backlog": "Beg Backlog",
            "new_demand": "New Demand",
            "capacity": "Capacity",
            "capacity_recommended": "Capacity (Rec)",
            "capacity_actual": "Capacity (Actual)",
            "end_backlog": "End Backlog",
            "end_backlog_recommended": "End Backlog (Rec)",
            "end_backlog_actual": "End Backlog (Actual)",
            "days_of_backlog": "Days of Backlog",
            "days_of_backlog_recommended": "Days Backlog (Rec)",
            "days_of_backlog_actual": "Days Backlog (Actual)",
            "alert_status": "Status",
            "hc_inbound": "HC Inbound",
            "hc_outbound": "HC Outbound",
            "hc_total": "HC Total",
            "hc_recommended": "(Rec)",
            "hc_actual": "(Actual)",
            "target_backlog": "Target Backlog",
            "forecast_col": "Forecast",
            "actual": "Actual",
            "error": "Abs Error",
            "pct_error": "% Error",
            "daily_cost_recommended": "Daily Cost (Rec)",
            "daily_cost_actual": "Daily Cost (Actual)",
            "daily_savings": "Daily Savings",
        },
        "alerts": {
            "healthy": "Healthy",
            "watch": "Watch",
            "critical": "Critical",
        },
        "labels": {
            "client": "Client",
            "report_date": "Report Date",
            "forecast_period": "Forecast Period",
            "wape": "WAPE",
            "mape": "MAPE",
            "mae": "MAE",
            "avg_days_of_backlog": "Avg Days of Backlog",
            "peak_headcount": "Peak Headcount",
            "critical_alert_days": "Critical Alert Days",
            "trend": "Trend",
            "improving": "Improving",
            "stable": "Stable",
            "degrading": "Degrading",
            "no_accuracy_data": "No historical accuracy data available yet.",
            "outbound": "Outbound",
            "inbound": "Inbound",
            "total": "Total",
            "weekly_summary": "Weekly Summary",
            "avg_daily_hc": "Avg Daily HC",
            "total_hours": "Total Hours",
            "total_cost_recommended": "Total Cost (Rec)",
            "total_cost_actual": "Total Cost (Actual)",
            "total_savings": "Total Savings",
            "avg_daily_savings": "Avg Daily Savings",
            "cost_savings": "Cost Savings",
        },
    }

    with open(locales_dir / "en.yaml", "w") as f:
        yaml.dump(en_locale, f)

    return str(locales_dir)


class TestRunPipeline:
    """Integration tests for run_pipeline with LightGBM forecasting."""

    def test_full_pipeline_produces_markdown(self, tmp_path):
        """Test that the full pipeline produces a report.md file."""
        config_path = _make_config_file(tmp_path)
        data_path = _make_data_file(tmp_path)
        locales_dir = _make_locale_files(tmp_path)
        output_dir = str(tmp_path / "output")
        log_dir = str(tmp_path / "logs")

        from hireplanner.pipeline.runner import run_pipeline

        result = run_pipeline(
            client_config_path=config_path,
            data_path=data_path,
            output_dir=output_dir,
            log_dir=log_dir,
            locales_dir=locales_dir,
        )

        assert result.endswith("report.md")
        assert Path(result).exists()

        # Verify figures directory exists with PNGs
        figures_dir = Path(result).parent / "figures"
        assert figures_dir.exists()
        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) > 0

    def test_dry_run_does_not_produce_file(self, tmp_path):
        """Test that dry_run validates but doesn't generate report."""
        config_path = _make_config_file(tmp_path)
        data_path = _make_data_file(tmp_path)
        locales_dir = _make_locale_files(tmp_path)

        from hireplanner.pipeline.runner import run_pipeline

        result = run_pipeline(
            client_config_path=config_path,
            data_path=data_path,
            output_dir=str(tmp_path / "output"),
            log_dir=str(tmp_path / "logs"),
            locales_dir=locales_dir,
            dry_run=True,
        )

        assert result == ""

    def test_pipeline_saves_forecast_for_accuracy(self, tmp_path):
        """Test that the pipeline saves the forecast for future comparison."""
        config_path = _make_config_file(tmp_path)
        data_path = _make_data_file(tmp_path)
        locales_dir = _make_locale_files(tmp_path)
        log_dir = str(tmp_path / "logs")

        from hireplanner.pipeline.runner import run_pipeline

        run_pipeline(
            client_config_path=config_path,
            data_path=data_path,
            output_dir=str(tmp_path / "output"),
            log_dir=log_dir,
            locales_dir=locales_dir,
        )

        # Check that a forecast file was saved
        log_path = Path(log_dir)
        forecast_files = list(log_path.glob("*_forecast_*.csv"))
        assert len(forecast_files) == 1

    def test_invalid_config_raises_error(self, tmp_path):
        """Test that invalid config causes a clear error."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("client_name: ''")

        from hireplanner.pipeline.runner import run_pipeline
        from hireplanner.config.client_config import ConfigError

        with pytest.raises(ConfigError):
            run_pipeline(
                client_config_path=str(bad_config),
                data_path="nonexistent.csv",
                output_dir=str(tmp_path / "output"),
            )

    def test_pipeline_fallback_on_lgbm_failure(self, tmp_path):
        """Test that pipeline falls back to naive forecast when LightGBM fails."""
        config_path = _make_config_file(tmp_path)
        data_path = _make_data_file(tmp_path)
        locales_dir = _make_locale_files(tmp_path)

        from hireplanner.pipeline.runner import run_pipeline

        # Force LightGBM to fail
        with patch(
            "hireplanner.forecasting.lightgbm_model.LightGBMForecaster.forecast",
            side_effect=RuntimeError("forced failure"),
        ):
            result = run_pipeline(
                client_config_path=config_path,
                data_path=data_path,
                output_dir=str(tmp_path / "output"),
                log_dir=str(tmp_path / "logs"),
                locales_dir=locales_dir,
            )

        assert result.endswith("report.md")
        assert Path(result).exists()


class TestCLI:
    """Tests for the CLI argument parser."""

    def test_main_requires_config_and_data(self):
        """Test that CLI requires --config and --data args."""
        from hireplanner.pipeline.runner import main

        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["hireplanner"]):
                main()
        assert exc_info.value.code == 2  # argparse error

    def test_main_with_missing_file_exits_1(self, tmp_path):
        """Test that CLI exits with code 1 on missing data file."""
        config_path = _make_config_file(tmp_path)

        from hireplanner.pipeline.runner import main

        with pytest.raises(SystemExit) as exc_info:
            with patch(
                "sys.argv",
                ["hireplanner", "--config", config_path, "--data", "nonexistent.csv"],
            ):
                main()
        assert exc_info.value.code == 1
