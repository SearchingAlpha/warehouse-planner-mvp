"""Data cleaning pipeline for warehouse volume data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to an ingested volume DataFrame.

    Steps
    -----
    1. Fill date gaps — complete daily range, interpolate short gaps (<3 days),
       forward-fill longer ones.
    2. Clip negative values to 0 for all numeric columns.
    3. Flag outliers — values >3 std devs from 21-day rolling median.
       Adds ``outbound_outlier`` and/or ``inbound_outlier`` boolean columns.
    4. Add calendar features: ``day_of_week``, ``week_of_year``, ``is_weekend``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date`` column and at least one of ``outbound`` / ``inbound``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.copy()
    df = _fill_date_gaps(df)
    df = _clip_negatives(df)
    df = _flag_outliers(df)
    df = _add_calendar_features(df)
    return df


def _fill_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete daily date range and fill missing values."""
    df = df.set_index("date")

    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range)
    df.index.name = "date"

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        # Identify groups of consecutive NaNs
        is_nan = df[col].isna()
        if not is_nan.any():
            continue

        # Label consecutive NaN groups
        groups = (~is_nan).cumsum()
        nan_groups = groups[is_nan]

        # For each NaN group, count its length
        group_lengths = nan_groups.groupby(nan_groups).transform("count")

        # Short gaps (<3 days): interpolate
        short_mask = is_nan & (group_lengths < 3)
        # Long gaps (>=3 days): will be forward-filled

        # First interpolate the whole column (handles short gaps)
        interpolated = df[col].interpolate(method="linear")

        # For long gaps, use forward-fill instead
        long_mask = is_nan & (group_lengths >= 3)
        ffilled = df[col].ffill()

        # Start with the original, apply interpolation for short, ffill for long
        result = df[col].copy()
        result[short_mask] = interpolated[short_mask]
        result[long_mask] = ffilled[long_mask]

        # If there are still NaNs at the start (no value to ffill from), backfill
        result = result.bfill()

        df[col] = result

    df = df.reset_index()
    return df


def _clip_negatives(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative values to 0 for all numeric columns."""
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    return df


def _flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag values >3 std devs from 21-day rolling median."""
    for col in ("outbound", "inbound"):
        flag_col = f"{col}_outlier"
        if col not in df.columns:
            continue

        rolling_median = df[col].rolling(window=21, center=True, min_periods=1).median()
        rolling_std = df[col].rolling(window=21, center=True, min_periods=1).std()

        # Avoid flagging when std is 0 or NaN
        rolling_std = rolling_std.replace(0, np.nan)

        deviation = (df[col] - rolling_median).abs()
        df[flag_col] = deviation > (3 * rolling_std)
        # Where std was NaN, mark as not outlier
        df[flag_col] = df[flag_col].fillna(False)

    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, week_of_year, and is_weekend columns."""
    df["day_of_week"] = df["date"].dt.dayofweek  # 0 = Monday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6])
    return df
