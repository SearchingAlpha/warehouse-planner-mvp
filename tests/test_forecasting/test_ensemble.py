"""Tests for ensemble / model selection logic."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from hireplanner.forecasting.ensemble import (
    blend_forecasts,
    build_forecast_df,
    select_best_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_forecast(p10, p50, p90, dates=None):
    """Helper to build a forecast result dict."""
    result = {
        "p10": np.array(p10, dtype=float),
        "p50": np.array(p50, dtype=float),
        "p90": np.array(p90, dtype=float),
    }
    if dates is not None:
        result["dates"] = np.array(dates, dtype="datetime64[ns]")
    return result


# ---------------------------------------------------------------------------
# select_best_model tests
# ---------------------------------------------------------------------------

class TestSelectBestModel:
    """Tests for select_best_model."""

    def test_selects_lower_wape_model_with_actuals(self):
        """When actuals are provided, the model with lower WAPE is selected."""
        actuals = np.array([10.0, 20.0, 30.0])
        # model_a predicts perfectly, model_b is off
        models = {
            "model_a": _make_forecast([8, 16, 24], [10, 20, 30], [12, 24, 36]),
            "model_b": _make_forecast([5, 10, 15], [50, 60, 70], [55, 65, 75]),
        }
        with patch("hireplanner.metrics.evaluation.wape") as mock_wape:
            # model_a has wape=0.0, model_b has wape=2.0
            mock_wape.side_effect = lambda a, p: 0.0 if p[0] == 10.0 else 2.0
            best = select_best_model(models, actuals=actuals)
        assert best == "model_a"

    def test_defaults_to_lightgbm_without_actuals(self):
        """Without actuals, 'lightgbm' is preferred if present."""
        models = {
            "other": _make_forecast([1], [2], [3]),
            "lightgbm": _make_forecast([1], [2], [3]),
        }
        best = select_best_model(models, actuals=None)
        assert best == "lightgbm"

    def test_falls_back_to_available_model_without_actuals(self):
        """Without actuals and no 'lightgbm', returns first available model."""
        models = {
            "other_model": _make_forecast([1], [2], [3]),
        }
        best = select_best_model(models, actuals=None)
        assert best == "other_model"


# ---------------------------------------------------------------------------
# blend_forecasts tests
# ---------------------------------------------------------------------------

class TestBlendForecasts:
    """Tests for blend_forecasts."""

    def test_equal_weights_blend(self):
        """Equal-weight blend averages the forecasts."""
        models = {
            "a": _make_forecast([10, 20], [100, 200], [1000, 2000]),
            "b": _make_forecast([30, 40], [300, 400], [3000, 4000]),
        }
        result = blend_forecasts(models)
        np.testing.assert_allclose(result["p10"], [20.0, 30.0])
        np.testing.assert_allclose(result["p50"], [200.0, 300.0])
        np.testing.assert_allclose(result["p90"], [2000.0, 3000.0])

    def test_custom_weights_blend(self):
        """Custom weights produce a weighted average."""
        models = {
            "a": _make_forecast([0, 0], [10, 10], [20, 20]),
            "b": _make_forecast([0, 0], [30, 30], [40, 40]),
        }
        # 75% weight on model b
        result = blend_forecasts(models, weights={"a": 0.25, "b": 0.75})
        np.testing.assert_allclose(result["p50"], [25.0, 25.0])

    def test_single_model_returned_unchanged(self):
        """With only one model, its result is returned as-is."""
        original = _make_forecast([1, 2], [3, 4], [5, 6])
        models = {"only": original}
        result = blend_forecasts(models)
        assert result is original

    def test_all_values_non_negative(self):
        """Blended forecast values should never be negative."""
        models = {
            "a": _make_forecast([-10, -5], [-3, -1], [0, 1]),
            "b": _make_forecast([-20, -15], [-8, -4], [-2, 0]),
        }
        result = blend_forecasts(models)
        for key in ("p10", "p50", "p90"):
            assert (result[key] >= 0).all(), f"{key} contains negative values"

    def test_dates_carried_from_model(self):
        """Dates from the first model that has them are carried through."""
        dates = pd.date_range("2026-01-01", periods=3)
        models = {
            "a": _make_forecast([1, 2, 3], [4, 5, 6], [7, 8, 9]),
            "b": _make_forecast(
                [1, 2, 3], [4, 5, 6], [7, 8, 9], dates=dates
            ),
        }
        result = blend_forecasts(models)
        assert "dates" in result
        np.testing.assert_array_equal(result["dates"], dates.values)


# ---------------------------------------------------------------------------
# build_forecast_df tests
# ---------------------------------------------------------------------------

class TestBuildForecastDf:
    """Tests for build_forecast_df."""

    def test_correct_shape_and_columns(self):
        """Output DataFrame has the correct shape and expected columns."""
        dates = pd.date_range("2026-01-01", periods=7)
        forecast = _make_forecast(
            list(range(7)),
            list(range(10, 17)),
            list(range(20, 27)),
        )
        df = build_forecast_df(forecast, dates, flow="inbound")
        assert df.shape == (7, 5)
        assert set(df.columns) == {
            "date", "flow", "forecast_p50", "forecast_p10", "forecast_p90",
        }
        assert (df["flow"] == "inbound").all()

    def test_date_column_is_datetime(self):
        """The date column is properly converted to datetime."""
        dates = pd.date_range("2026-03-01", periods=3)
        forecast = _make_forecast([1, 2, 3], [4, 5, 6], [7, 8, 9])
        df = build_forecast_df(forecast, dates, flow="outbound")
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
