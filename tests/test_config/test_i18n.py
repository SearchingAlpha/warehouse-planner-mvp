"""Tests for the i18n (multi-language) module."""
from pathlib import Path

import pytest
import yaml

from hireplanner.config.i18n import load_locale, t


LOCALES_DIR = Path(__file__).resolve().parents[2] / "configs" / "locales"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadLocale:
    def test_load_english(self):
        locale = load_locale("en", LOCALES_DIR)
        assert "tabs" in locale
        assert locale["tabs"]["executive_summary"] == "Executive Summary"

    def test_load_spanish(self):
        locale = load_locale("es", LOCALES_DIR)
        assert "tabs" in locale
        assert locale["tabs"]["executive_summary"] == "Resumen Ejecutivo"

    def test_fallback_to_english_for_unknown_language(self):
        locale = load_locale("fr", LOCALES_DIR)
        # Should fall back to English
        assert locale["tabs"]["executive_summary"] == "Executive Summary"

    def test_raises_when_no_files_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_locale("en", tmp_path)


class TestTranslateFunction:
    @pytest.fixture
    def en_locale(self):
        return load_locale("en", LOCALES_DIR)

    def test_simple_key_lookup(self, en_locale):
        # Top-level section lookup returns a dict, but a leaf key returns string
        assert t("labels.client", en_locale) == "Client"

    def test_nested_dotted_key(self, en_locale):
        assert t("tabs.executive_summary", en_locale) == "Executive Summary"
        assert t("headers.date", en_locale) == "Date"
        assert t("alerts.critical", en_locale) == "Critical"

    def test_returns_key_when_not_found(self, en_locale):
        assert t("nonexistent.key.path", en_locale) == "nonexistent.key.path"


class TestLocaleCompleteness:
    def test_all_required_keys_exist_in_both_locales(self):
        en = load_locale("en", LOCALES_DIR)
        es = load_locale("es", LOCALES_DIR)

        def collect_keys(d, prefix=""):
            """Recursively collect all leaf keys as dotted paths."""
            keys = set()
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    keys.update(collect_keys(v, full_key))
                else:
                    keys.add(full_key)
            return keys

        en_keys = collect_keys(en)
        es_keys = collect_keys(es)

        missing_in_es = en_keys - es_keys
        missing_in_en = es_keys - en_keys

        assert not missing_in_es, f"Keys missing in es.yaml: {missing_in_es}"
        assert not missing_in_en, f"Keys missing in en.yaml: {missing_in_en}"
