"""Data loading with auto-detection of column names for warehouse volume data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Raised when data cannot be loaded or required columns are missing."""


# Mapping from canonical name -> list of accepted aliases (all lowercase).
_DATE_ALIASES = ["date", "ds", "timestamp", "fecha"]
_OUTBOUND_ALIASES = ["outbound", "ob", "shipped", "enviado"]
_INBOUND_ALIASES = ["inbound", "ib", "received", "recibido"]


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    """Return the first column name (original casing) that matches any alias."""
    lower_map = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file and normalise column names.

    Auto-detects date, outbound, and inbound columns from common aliases.
    Returns a DataFrame with standardised column names sorted by date ascending.

    Parameters
    ----------
    path : str | Path
        Path to the data file (.csv, .xlsx, .xls).

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``date`` and one of ``outbound`` / ``inbound``.

    Raises
    ------
    DataLoadError
        If the file is not found, has an unsupported format, or is missing
        required columns.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataLoadError(f"Failed to read CSV file: {exc}") from exc
    elif suffix in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            raise DataLoadError(f"Failed to read Excel file: {exc}") from exc
    else:
        raise DataLoadError(
            f"Unsupported file format '{suffix}'. Supported formats: .csv, .xlsx, .xls"
        )

    if df.empty:
        raise DataLoadError("File is empty or contains no data rows.")

    # --- Auto-detect and rename columns ---
    rename_map: dict[str, str] = {}

    date_col = _find_column(df.columns.tolist(), _DATE_ALIASES)
    if date_col is None:
        raise DataLoadError(
            f"No date column found. Expected one of: {_DATE_ALIASES}. "
            f"Found columns: {df.columns.tolist()}"
        )
    rename_map[date_col] = "date"

    outbound_col = _find_column(df.columns.tolist(), _OUTBOUND_ALIASES)
    if outbound_col is not None:
        rename_map[outbound_col] = "outbound"

    inbound_col = _find_column(df.columns.tolist(), _INBOUND_ALIASES)
    if inbound_col is not None:
        rename_map[inbound_col] = "inbound"

    if outbound_col is None and inbound_col is None:
        raise DataLoadError(
            "No volume columns found. Expected at least one of: "
            f"outbound aliases {_OUTBOUND_ALIASES} or inbound aliases {_INBOUND_ALIASES}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df.rename(columns=rename_map)

    # Parse date column
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as exc:
        raise DataLoadError(f"Failed to parse date column: {exc}") from exc

    df = df.sort_values("date").reset_index(drop=True)
    return df
