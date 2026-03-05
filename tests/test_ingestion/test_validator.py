"""Tests for hireplanner.ingestion.validator."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hireplanner.ingestion.validator import ValidationError, validate_data, validate_data_strict


def _make_df(days: int = 400, include_outlier_cols: bool = False) -> pd.DataFrame:
    """Create a test DataFrame with the given number of days."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "outbound": np.maximum(0, 1000 + np.random.normal(0, 100, days)).astype(int),
            "inbound": np.maximum(0, 500 + np.random.normal(0, 50, days)).astype(int),
        }
    )
    if include_outlier_cols:
        df["outbound_outlier"] = False
        df["inbound_outlier"] = False
    return df


# ---------------------------------------------------------------------------
# validate_data (warnings)
# ---------------------------------------------------------------------------


class TestValidateData:
    def test_valid_data_no_warnings(self):
        """24+ month data with no issues should produce no warnings."""
        df = _make_df(days=740, include_outlier_cols=True)
        warnings = validate_data(df)
        assert warnings == []

    def test_short_history_warning(self):
        """Data between 365 and 730 days should produce a recommendation warning."""
        df = _make_df(days=400, include_outlier_cols=True)
        warnings = validate_data(df)
        assert any("Recommend" in w or "Short history" in w for w in warnings)

    def test_insufficient_history_error_message(self):
        """Data shorter than 365 days should produce an insufficient history warning."""
        df = _make_df(days=100)
        warnings = validate_data(df)
        assert any("Insufficient" in w for w in warnings)

    def test_missing_volume_columns(self):
        """DataFrame with no outbound/inbound should warn."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=400), "other": range(400)})
        warnings = validate_data(df)
        assert any("volume" in w.lower() for w in warnings)

    def test_high_missing_data_warning(self):
        """More than 10% NaN in volume columns should warn."""
        df = _make_df(days=400)
        # Set >10% of both volume columns to NaN so combined ratio > 10%
        nan_count = int(len(df) * 0.15)
        df.loc[:nan_count, "outbound"] = np.nan
        df.loc[:nan_count, "inbound"] = np.nan
        warnings = validate_data(df)
        assert any("missing" in w.lower() for w in warnings)

    def test_high_outlier_rate_warning(self):
        """If >5% of points are flagged as outliers, should warn."""
        df = _make_df(days=400, include_outlier_cols=True)
        # Flag 10% as outliers
        n_outliers = int(len(df) * 0.10)
        df.loc[:n_outliers, "outbound_outlier"] = True
        warnings = validate_data(df)
        assert any("outlier" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# validate_data_strict
# ---------------------------------------------------------------------------


class TestValidateDataStrict:
    def test_valid_data_passes(self):
        """Valid data should not raise."""
        df = _make_df(days=400)
        validate_data_strict(df)  # Should not raise

    def test_too_short_raises(self):
        """Data shorter than 365 days should raise ValidationError."""
        df = _make_df(days=100)
        with pytest.raises(ValidationError, match="Insufficient history"):
            validate_data_strict(df)

    def test_too_much_missing_raises(self):
        """More than 10% missing data should raise ValidationError."""
        df = _make_df(days=400)
        nan_count = int(len(df) * 0.15)
        df.loc[:nan_count, "outbound"] = np.nan
        df.loc[:nan_count, "inbound"] = np.nan
        with pytest.raises(ValidationError, match="missing"):
            validate_data_strict(df)

    def test_no_volume_columns_raises(self):
        """DataFrame with no volume columns should raise."""
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=400), "other": range(400)})
        with pytest.raises(ValidationError, match="volume"):
            validate_data_strict(df)
