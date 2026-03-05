"""Tests for hireplanner.ingestion.cleaner."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from hireplanner.ingestion.cleaner import clean_data


def _make_df(
    start: str = "2024-01-01",
    days: int = 30,
    outbound_base: float = 1000,
    inbound_base: float = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a simple daily DataFrame for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "outbound": (outbound_base + np.random.normal(0, 50, days)).astype(int),
            "inbound": (inbound_base + np.random.normal(0, 30, days)).astype(int),
        }
    )


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------


class TestGapFilling:
    def test_short_gap_interpolated(self):
        """Gaps of 1-2 days should be linearly interpolated."""
        df = _make_df(days=10)
        # Remove days at index 3 and 4 (2-day gap)
        df = df.drop([3, 4]).reset_index(drop=True)
        assert len(df) == 8

        cleaned = clean_data(df)
        # Should have 10 days again
        assert len(cleaned) == 10
        # The interpolated values should not be NaN
        assert cleaned["outbound"].isna().sum() == 0

    def test_long_gap_forward_filled(self):
        """Gaps of >=3 days should be forward-filled."""
        df = _make_df(days=15)
        # Remove days at index 5, 6, 7 (3-day gap)
        df = df.drop([5, 6, 7]).reset_index(drop=True)

        cleaned = clean_data(df)
        assert len(cleaned) == 15
        assert cleaned["outbound"].isna().sum() == 0

    def test_no_gaps_unchanged(self):
        """A complete daily series should pass through unchanged in length."""
        df = _make_df(days=30)
        cleaned = clean_data(df)
        assert len(cleaned) == 30


# ---------------------------------------------------------------------------
# Negative clipping
# ---------------------------------------------------------------------------


class TestNegativeClipping:
    def test_negatives_clipped_to_zero(self):
        """Negative values in volume columns should be clipped to 0."""
        df = _make_df(days=5)
        df.loc[0, "outbound"] = -100
        df.loc[2, "inbound"] = -50

        cleaned = clean_data(df)
        assert cleaned["outbound"].min() >= 0
        assert cleaned["inbound"].min() >= 0

    def test_zero_values_preserved(self):
        """Zero values should remain zero."""
        df = _make_df(days=5)
        df.loc[0, "outbound"] = 0

        cleaned = clean_data(df)
        assert cleaned.loc[0, "outbound"] == 0


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------


class TestOutlierFlagging:
    def test_outlier_columns_created(self):
        """Outlier flag columns should be present in output."""
        df = _make_df(days=60)
        cleaned = clean_data(df)
        assert "outbound_outlier" in cleaned.columns
        assert "inbound_outlier" in cleaned.columns
        assert cleaned["outbound_outlier"].dtype == bool

    def test_extreme_value_flagged(self):
        """An extreme spike should be flagged as an outlier."""
        df = _make_df(days=60, outbound_base=1000)
        # Inject a huge spike in the middle
        df.loc[30, "outbound"] = 50000

        cleaned = clean_data(df)
        assert cleaned.loc[30, "outbound_outlier"] is True or cleaned.loc[30, "outbound_outlier"]

    def test_normal_values_not_flagged(self):
        """Normal values should not be flagged."""
        df = _make_df(days=60, outbound_base=1000)
        cleaned = clean_data(df)
        # Most values should not be outliers
        assert cleaned["outbound_outlier"].sum() < len(cleaned) * 0.2


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


class TestCalendarFeatures:
    def test_calendar_columns_added(self):
        """Calendar feature columns should be present."""
        df = _make_df(days=10)
        cleaned = clean_data(df)
        assert "day_of_week" in cleaned.columns
        assert "week_of_year" in cleaned.columns
        assert "is_weekend" in cleaned.columns

    def test_day_of_week_values(self):
        """day_of_week should be 0-6 with Monday=0."""
        df = _make_df(start="2024-01-01", days=7)  # Mon 2024-01-01
        cleaned = clean_data(df)
        assert cleaned.iloc[0]["day_of_week"] == 0  # Monday
        assert cleaned.iloc[5]["day_of_week"] == 5  # Saturday
        assert cleaned.iloc[6]["day_of_week"] == 6  # Sunday

    def test_is_weekend(self):
        """Saturday and Sunday should be flagged as weekend."""
        df = _make_df(start="2024-01-01", days=7)
        cleaned = clean_data(df)
        weekend_mask = cleaned["is_weekend"]
        assert not weekend_mask.iloc[0]  # Monday
        assert weekend_mask.iloc[5]  # Saturday
        assert weekend_mask.iloc[6]  # Sunday
