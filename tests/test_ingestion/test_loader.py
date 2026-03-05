"""Tests for hireplanner.ingestion.loader."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from hireplanner.ingestion.loader import DataLoadError, load_data


def _write_csv(path, content: str):
    """Helper to write CSV content to a file."""
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestLoadCSV:
    def test_load_basic_csv(self, tmp_path):
        """Load a well-formed CSV with standard column names."""
        csv = tmp_path / "data.csv"
        _write_csv(
            csv,
            "date,outbound,inbound\n"
            "2024-01-01,100,50\n"
            "2024-01-02,110,55\n"
            "2024-01-03,120,60\n",
        )
        df = load_data(csv)

        assert list(df.columns[:3]) == ["date", "outbound", "inbound"]
        assert len(df) == 3
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        # Sorted ascending
        assert df["date"].is_monotonic_increasing

    def test_load_csv_sorted_descending_is_reordered(self, tmp_path):
        """CSV with dates in descending order should be sorted ascending."""
        csv = tmp_path / "desc.csv"
        _write_csv(
            csv,
            "date,outbound,inbound\n"
            "2024-01-03,120,60\n"
            "2024-01-01,100,50\n"
            "2024-01-02,110,55\n",
        )
        df = load_data(csv)
        assert df["date"].is_monotonic_increasing
        assert df.iloc[0]["outbound"] == 100


class TestLoadExcel:
    def test_load_xlsx(self, tmp_path):
        """Load a basic Excel file."""
        xlsx = tmp_path / "data.xlsx"
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "outbound": [100, 110, 120, 130, 140],
                "inbound": [50, 55, 60, 65, 70],
            }
        ).to_excel(xlsx, index=False, engine="openpyxl")

        df = load_data(xlsx)
        assert len(df) == 5
        assert "outbound" in df.columns


# ---------------------------------------------------------------------------
# Auto-detect column aliases
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_detect_ds_as_date(self, tmp_path):
        """'ds' should be detected as the date column."""
        csv = tmp_path / "ds.csv"
        _write_csv(csv, "ds,outbound\n2024-01-01,100\n2024-01-02,110\n")
        df = load_data(csv)
        assert "date" in df.columns

    def test_detect_timestamp_as_date(self, tmp_path):
        """'Timestamp' (mixed case) should be detected as the date column."""
        csv = tmp_path / "ts.csv"
        _write_csv(csv, "Timestamp,OB\n2024-01-01,100\n2024-01-02,110\n")
        df = load_data(csv)
        assert "date" in df.columns
        assert "outbound" in df.columns

    def test_detect_fecha_as_date(self, tmp_path):
        """'Fecha' (Spanish) should be detected as the date column."""
        csv = tmp_path / "es.csv"
        _write_csv(csv, "Fecha,Enviado,Recibido\n2024-01-01,100,50\n2024-01-02,110,55\n")
        df = load_data(csv)
        assert "date" in df.columns
        assert "outbound" in df.columns
        assert "inbound" in df.columns

    def test_detect_shipped_and_received(self, tmp_path):
        """'shipped' and 'received' should map to outbound/inbound."""
        csv = tmp_path / "alt.csv"
        _write_csv(csv, "date,shipped,received\n2024-01-01,100,50\n")
        df = load_data(csv)
        assert "outbound" in df.columns
        assert "inbound" in df.columns

    def test_outbound_only_is_valid(self, tmp_path):
        """A file with only outbound (no inbound) is valid."""
        csv = tmp_path / "ob.csv"
        _write_csv(csv, "date,OB\n2024-01-01,100\n2024-01-02,200\n")
        df = load_data(csv)
        assert "outbound" in df.columns
        assert "inbound" not in df.columns


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DataLoadError, match="File not found"):
            load_data(tmp_path / "nonexistent.csv")

    def test_unsupported_format(self, tmp_path):
        txt = tmp_path / "data.txt"
        txt.write_text("a,b\n1,2\n")
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            load_data(txt)

    def test_missing_date_column(self, tmp_path):
        csv = tmp_path / "nodate.csv"
        _write_csv(csv, "col_a,col_b\n1,2\n3,4\n")
        with pytest.raises(DataLoadError, match="No date column found"):
            load_data(csv)

    def test_missing_volume_columns(self, tmp_path):
        csv = tmp_path / "novol.csv"
        _write_csv(csv, "date,category\n2024-01-01,A\n2024-01-02,B\n")
        with pytest.raises(DataLoadError, match="No volume columns found"):
            load_data(csv)
