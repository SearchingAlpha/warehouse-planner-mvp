from dataclasses import dataclass, field
from pathlib import Path
import yaml

class ConfigError(Exception):
    pass

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
    language: str = "en"
    forecast_horizon: int = 28

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
        # Language
        if self.language not in ("es", "en"):
            raise ConfigError(f"language must be 'es' or 'en', got '{self.language}'")
        # Horizon
        if self.forecast_horizon < 1:
            raise ConfigError("forecast_horizon must be at least 1")

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
            language=str(raw.get("language", "en")),
            forecast_horizon=int(raw.get("forecast_horizon", 28)),
        )
    except (ValueError, TypeError) as e:
        raise ConfigError(f"Invalid config value: {e}")

    config.validate()
    return config
