from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

class ConfigError(Exception):
    pass

_DEFAULT_SHIFT_PATTERNS = [{"name": "default", "days": [0, 1, 2, 3, 4, 5, 6]}]

@dataclass
class ClientConfig:
    client_name: str
    active_flows: list[str]
    productivity_inbound: float
    productivity_outbound: float
    hours_per_shift: int = 8
    overhead_buffer: float = 0.15
    backlog_threshold_watch: float = 1.0
    backlog_threshold_critical: float = 2.0
    initial_backlog_outbound: int = 0
    initial_backlog_inbound: int = 0
    target_backlog_ratio: float = 0.35
    current_staffing_outbound: int | list[int] = 0
    current_staffing_inbound: int | list[int] = 0
    shift_patterns: list[dict[str, Any]] | None = None
    language: str = "en"
    forecast_horizon: int = 28
    cost_per_hour: float = 0.0

    def get_shift_patterns(self) -> list[dict[str, Any]]:
        """Return effective shift patterns (default: single all-week rotation)."""
        if self.shift_patterns is None:
            return _DEFAULT_SHIFT_PATTERNS
        return self.shift_patterns

    def has_actual_staffing(self, flow: str) -> bool:
        """Return True if actual staffing is configured for *flow*."""
        val = getattr(self, f"current_staffing_{flow}")
        if isinstance(val, list):
            return any(v > 0 for v in val)
        return val > 0

    def validate(self):
        """Validate config values. Raises ConfigError on invalid values."""
        # Required fields
        if not self.client_name or not self.client_name.strip():
            raise ConfigError("client_name is required")
        if not self.active_flows:
            raise ConfigError("active_flows must contain at least one flow")
        for flow in self.active_flows:
            if flow not in ("outbound", "inbound"):
                raise ConfigError(f"Invalid flow '{flow}'. Must be 'outbound' or 'inbound'")
        # Productivity must be positive
        if self.productivity_inbound <= 0:
            raise ConfigError("productivity_inbound must be positive")
        if self.productivity_outbound <= 0:
            raise ConfigError("productivity_outbound must be positive")
        # Hours and buffer
        if self.hours_per_shift <= 0:
            raise ConfigError("hours_per_shift must be positive")
        if not 0 <= self.overhead_buffer <= 1:
            raise ConfigError("overhead_buffer must be between 0 and 1")
        # Thresholds
        if self.backlog_threshold_watch < 0:
            raise ConfigError("backlog_threshold_watch must be non-negative")
        if self.backlog_threshold_critical < 0:
            raise ConfigError("backlog_threshold_critical must be non-negative")
        if self.backlog_threshold_critical <= self.backlog_threshold_watch:
            raise ConfigError("backlog_threshold_critical must be greater than backlog_threshold_watch")
        # Initial backlogs
        if self.initial_backlog_outbound < 0:
            raise ConfigError("initial_backlog_outbound must be non-negative")
        if self.initial_backlog_inbound < 0:
            raise ConfigError("initial_backlog_inbound must be non-negative")
        # Target backlog ratio
        if not 0 <= self.target_backlog_ratio <= 1:
            raise ConfigError("target_backlog_ratio must be between 0 and 1")
        # Current staffing
        n_patterns = len(self.get_shift_patterns())
        for side in ("outbound", "inbound"):
            val = getattr(self, f"current_staffing_{side}")
            if isinstance(val, list):
                if len(val) != n_patterns:
                    raise ConfigError(
                        f"current_staffing_{side} has {len(val)} values "
                        f"but there are {n_patterns} shift patterns"
                    )
                for v in val:
                    if v < 0:
                        raise ConfigError(f"current_staffing_{side} values must be non-negative")
            else:
                if val < 0:
                    raise ConfigError(f"current_staffing_{side} must be non-negative")
        # Shift patterns
        if self.shift_patterns is not None:
            if not isinstance(self.shift_patterns, list) or len(self.shift_patterns) == 0:
                raise ConfigError("shift_patterns must be a non-empty list")
            for i, pat in enumerate(self.shift_patterns):
                if not isinstance(pat, dict):
                    raise ConfigError(f"shift_patterns[{i}] must be a mapping")
                if "days" not in pat:
                    raise ConfigError(f"shift_patterns[{i}] must have a 'days' key")
                days = pat["days"]
                if not isinstance(days, list) or len(days) == 0:
                    raise ConfigError(f"shift_patterns[{i}].days must be a non-empty list")
                for d in days:
                    if not isinstance(d, int) or d < 0 or d > 6:
                        raise ConfigError(
                            f"shift_patterns[{i}].days values must be integers 0-6 "
                            f"(0=Monday, 6=Sunday), got {d!r}"
                        )
                if len(days) != len(set(days)):
                    raise ConfigError(f"shift_patterns[{i}].days must not contain duplicates")
        # Language
        if self.language not in ("es", "en"):
            raise ConfigError(f"language must be 'es' or 'en', got '{self.language}'")
        # Cost per hour
        if self.cost_per_hour < 0:
            raise ConfigError("cost_per_hour must be non-negative")
        # Horizon
        if self.forecast_horizon < 1:
            raise ConfigError("forecast_horizon must be at least 1")


def _parse_staffing(raw_val) -> int | list[int]:
    """Parse a current_staffing value: int or list of ints."""
    if isinstance(raw_val, list):
        return [int(v) for v in raw_val]
    return int(raw_val)


def load_client_config(path: str | Path) -> ClientConfig:
    """Load and validate a client config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")

    # Required fields
    required = ["client_name", "active_flows", "productivity_inbound", "productivity_outbound"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Missing required config fields: {', '.join(missing)}")

    # Parse shift_patterns
    shift_patterns = raw.get("shift_patterns", None)
    if shift_patterns is not None:
        shift_patterns = list(shift_patterns)

    # Build config with defaults
    try:
        config = ClientConfig(
            client_name=str(raw["client_name"]),
            active_flows=list(raw["active_flows"]),
            productivity_inbound=float(raw["productivity_inbound"]),
            productivity_outbound=float(raw["productivity_outbound"]),
            hours_per_shift=int(raw.get("hours_per_shift", 8)),
            overhead_buffer=float(raw.get("overhead_buffer", 0.15)),
            backlog_threshold_watch=float(raw.get("backlog_threshold_watch", 1.0)),
            backlog_threshold_critical=float(raw.get("backlog_threshold_critical", 2.0)),
            initial_backlog_outbound=int(raw.get("initial_backlog_outbound", 0)),
            initial_backlog_inbound=int(raw.get("initial_backlog_inbound", 0)),
            target_backlog_ratio=float(raw.get("target_backlog_ratio", 0.35)),
            current_staffing_outbound=_parse_staffing(raw.get("current_staffing_outbound", 0)),
            current_staffing_inbound=_parse_staffing(raw.get("current_staffing_inbound", 0)),
            shift_patterns=shift_patterns,
            language=str(raw.get("language", "en")),
            forecast_horizon=int(raw.get("forecast_horizon", 28)),
            cost_per_hour=float(raw.get("cost_per_hour", 0.0)),
        )
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Invalid config value: {e}")

    config.validate()
    return config
