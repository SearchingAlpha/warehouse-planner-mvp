"""Tests for matplotlib chart generators."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hireplanner.reporting.matplotlib_charts import (
    save_forecast_chart,
    save_backlog_chart,
    save_headcount_chart,
    save_accuracy_chart,
)


@pytest.fixture
def dates():
    return pd.date_range("2026-03-01", periods=7, freq="D")


@pytest.fixture
def figures_dir(tmp_path):
    d = tmp_path / "figures"
    d.mkdir()
    return d


class TestSaveForecastChart:
    def test_creates_png(self, dates, figures_dir):
        path = figures_dir / "forecast.png"
        save_forecast_chart(
            dates,
            p10=np.array([4000] * 7),
            p50=np.array([5000] * 7),
            p90=np.array([6000] * 7),
            title="Test Forecast",
            path=path,
        )
        assert path.exists()
        assert path.stat().st_size > 0


class TestSaveBacklogChart:
    def test_creates_png_with_both_tracks(self, dates, figures_dir):
        path = figures_dir / "backlog.png"
        save_backlog_chart(
            dates,
            end_backlog_recommended=np.array([1000, 900, 800, 700, 600, 500, 400]),
            end_backlog_actual=np.array([1200, 1100, 1050, 1000, 950, 900, 850]),
            target_backlog_units=700.0,
            title="Test Backlog",
            path=path,
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_creates_png_without_actual(self, dates, figures_dir):
        path = figures_dir / "backlog_no_actual.png"
        save_backlog_chart(
            dates,
            end_backlog_recommended=np.array([1000, 900, 800, 700, 600, 500, 400]),
            end_backlog_actual=None,
            target_backlog_units=700.0,
            title="Test Backlog No Actual",
            path=path,
        )
        assert path.exists()
        assert path.stat().st_size > 0


class TestSaveHeadcountChart:
    def test_creates_png(self, dates, figures_dir):
        path = figures_dir / "headcount.png"
        save_headcount_chart(
            dates,
            hc_data={
                "Outbound (Rec)": np.array([10, 12, 11, 13, 10, 12, 11]),
                "Inbound (Rec)": np.array([5, 6, 5, 7, 5, 6, 5]),
            },
            title="Test HC",
            path=path,
        )
        assert path.exists()
        assert path.stat().st_size > 0


class TestSaveAccuracyChart:
    def test_creates_png(self, dates, figures_dir):
        path = figures_dir / "accuracy.png"
        save_accuracy_chart(
            dates,
            forecast=np.array([5000, 5100, 4900, 5200, 5050, 4950, 5150]),
            actual=np.array([5100, 5000, 5050, 5150, 5100, 5000, 5200]),
            title="Test Accuracy",
            path=path,
        )
        assert path.exists()
        assert path.stat().st_size > 0
