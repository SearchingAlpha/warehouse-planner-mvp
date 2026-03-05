"""Data validation for warehouse volume data."""

from __future__ import annotations

import pandas as pd


class ValidationError(Exception):
    """Raised when critical data validation checks fail."""


_MIN_DAYS_STRICT = 365  # 12 months minimum
_MIN_DAYS_RECOMMENDED = 730  # 24 months recommended
_MAX_MISSING_RATIO = 0.10  # 10%
_OUTLIER_WARN_RATIO = 0.05  # 5%


def validate_data(df: pd.DataFrame) -> list[str]:
    """Validate a cleaned DataFrame and return a list of warning messages.

    An empty list means the data passed all checks.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``date`` and volume columns.

    Returns
    -------
    list[str]
        Warning messages (may be empty).
    """
    warnings: list[str] = []

    # --- Check for at least one volume column ---
    has_outbound = "outbound" in df.columns
    has_inbound = "inbound" in df.columns
    if not has_outbound and not has_inbound:
        warnings.append("No volume columns (outbound/inbound) found in the data.")

    # --- Date range length ---
    if "date" in df.columns and len(df) > 0:
        date_range_days = (df["date"].max() - df["date"].min()).days + 1

        if date_range_days < _MIN_DAYS_STRICT:
            warnings.append(
                f"Insufficient history: {date_range_days} days found, "
                f"minimum {_MIN_DAYS_STRICT} days (12 months) required."
            )
        elif date_range_days < _MIN_DAYS_RECOMMENDED:
            warnings.append(
                f"Short history: {date_range_days} days found. "
                f"Recommend at least {_MIN_DAYS_RECOMMENDED} days (24 months) for best results."
            )

    # --- Missing data ratio ---
    volume_cols = [c for c in ("outbound", "inbound") if c in df.columns]
    if volume_cols:
        total_cells = len(df) * len(volume_cols)
        missing_cells = df[volume_cols].isna().sum().sum()
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            if missing_ratio > _MAX_MISSING_RATIO:
                warnings.append(
                    f"High missing data: {missing_ratio:.1%} of volume values are missing "
                    f"(threshold: {_MAX_MISSING_RATIO:.0%})."
                )

    # --- Outlier ratio ---
    for col in ("outbound", "inbound"):
        flag_col = f"{col}_outlier"
        if flag_col in df.columns:
            outlier_ratio = df[flag_col].sum() / len(df) if len(df) > 0 else 0
            if outlier_ratio > _OUTLIER_WARN_RATIO:
                warnings.append(
                    f"High outlier rate in {col}: {outlier_ratio:.1%} of data points flagged "
                    f"(threshold: {_OUTLIER_WARN_RATIO:.0%})."
                )

    return warnings


def validate_data_strict(df: pd.DataFrame) -> None:
    """Validate a cleaned DataFrame and raise on critical issues.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``date`` and volume columns.

    Raises
    ------
    ValidationError
        If any critical validation check fails.
    """
    errors: list[str] = []

    # --- At least one volume column ---
    has_outbound = "outbound" in df.columns
    has_inbound = "inbound" in df.columns
    if not has_outbound and not has_inbound:
        errors.append("No volume columns (outbound/inbound) found in the data.")

    # --- Minimum history ---
    if "date" in df.columns and len(df) > 0:
        date_range_days = (df["date"].max() - df["date"].min()).days + 1
        if date_range_days < _MIN_DAYS_STRICT:
            errors.append(
                f"Insufficient history: {date_range_days} days found, "
                f"minimum {_MIN_DAYS_STRICT} days (12 months) required."
            )

    # --- Missing data ---
    volume_cols = [c for c in ("outbound", "inbound") if c in df.columns]
    if volume_cols:
        total_cells = len(df) * len(volume_cols)
        missing_cells = df[volume_cols].isna().sum().sum()
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            if missing_ratio > _MAX_MISSING_RATIO:
                errors.append(
                    f"High missing data: {missing_ratio:.1%} of volume values are missing "
                    f"(threshold: {_MAX_MISSING_RATIO:.0%})."
                )

    if errors:
        raise ValidationError(
            "Critical validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
