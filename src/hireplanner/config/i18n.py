"""Multi-language support for report labels and headers."""
from pathlib import Path
from typing import Optional

import yaml


def load_locale(language: str, locales_dir: str | Path) -> dict:
    """Load a locale YAML file.

    Args:
        language: "en" or "es"
        locales_dir: Path to configs/locales/ directory

    Returns:
        Nested dict of translations
    """
    locales_dir = Path(locales_dir)
    path = locales_dir / f"{language}.yaml"

    if not path.exists():
        # Fallback to English
        path = locales_dir / "en.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No locale file found for '{language}' or 'en' in {locales_dir}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def t(key: str, locale: dict) -> str:
    """Translate a dotted key path. Falls back to the key itself if not found.

    Example: t("tabs.executive_summary", locale) -> "Executive Summary"
    """
    parts = key.split(".")
    current = locale
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return key  # Fallback: return the key itself
    return str(current)
