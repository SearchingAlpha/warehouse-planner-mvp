import pytest
import yaml

from hireplanner.config.client_config import ClientConfig, ConfigError, load_client_config


def _write_yaml(tmp_path, data):
    """Helper: write a dict as YAML to a temp file and return the path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


def _valid_data(**overrides):
    """Return a minimal valid config dict, with optional overrides."""
    base = {
        "client_name": "TestClient",
        "active_flows": ["outbound", "inbound"],
        "productivity_inbound": 120.0,
        "productivity_outbound": 95.0,
    }
    base.update(overrides)
    return base


class TestLoadValidConfig:
    def test_load_minimal_valid_config(self, tmp_path):
        path = _write_yaml(tmp_path, _valid_data())
        config = load_client_config(path)

        assert config.client_name == "TestClient"
        assert config.active_flows == ["outbound", "inbound"]
        assert config.productivity_inbound == 120.0
        assert config.productivity_outbound == 95.0

    def test_defaults_populated_correctly(self, tmp_path):
        path = _write_yaml(tmp_path, _valid_data())
        config = load_client_config(path)

        assert config.hours_per_shift == 8
        assert config.overhead_buffer == 0.15
        assert config.backlog_threshold_watch == 1.0
        assert config.backlog_threshold_critical == 2.0
        assert config.initial_backlog_outbound == 0
        assert config.initial_backlog_inbound == 0
        assert config.target_backlog_ratio == 0.35
        assert config.current_staffing_outbound == 0
        assert config.current_staffing_inbound == 0
        assert config.language == "en"
        assert config.forecast_horizon == 28
        assert config.cost_per_hour == 0.0

    def test_custom_values_override_defaults(self, tmp_path):
        data = _valid_data(
            hours_per_shift=10,
            overhead_buffer=0.20,
            language="es",
            forecast_horizon=14,
        )
        path = _write_yaml(tmp_path, data)
        config = load_client_config(path)

        assert config.hours_per_shift == 10
        assert config.overhead_buffer == 0.20
        assert config.language == "es"
        assert config.forecast_horizon == 14

    def test_only_outbound_flow(self, tmp_path):
        data = _valid_data(active_flows=["outbound"])
        path = _write_yaml(tmp_path, data)
        config = load_client_config(path)

        assert config.active_flows == ["outbound"]


class TestMissingRequiredFields:
    def test_missing_client_name(self, tmp_path):
        data = _valid_data()
        del data["client_name"]
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="Missing required config fields.*client_name"):
            load_client_config(path)

    def test_missing_active_flows(self, tmp_path):
        data = _valid_data()
        del data["active_flows"]
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="Missing required config fields.*active_flows"):
            load_client_config(path)

    def test_missing_productivity_fields(self, tmp_path):
        data = _valid_data()
        del data["productivity_inbound"]
        del data["productivity_outbound"]
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="Missing required config fields"):
            load_client_config(path)


class TestInvalidFlows:
    def test_invalid_flow_name(self, tmp_path):
        data = _valid_data(active_flows=["outbound", "picking"])
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="Invalid flow 'picking'"):
            load_client_config(path)

    def test_empty_active_flows(self, tmp_path):
        data = _valid_data(active_flows=[])
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="active_flows must contain at least one flow"):
            load_client_config(path)


class TestProductivityValidation:
    def test_zero_productivity_inbound(self, tmp_path):
        data = _valid_data(productivity_inbound=0)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="productivity_inbound must be positive"):
            load_client_config(path)

    def test_negative_productivity_outbound(self, tmp_path):
        data = _valid_data(productivity_outbound=-10)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="productivity_outbound must be positive"):
            load_client_config(path)


class TestOverheadBuffer:
    def test_overhead_buffer_above_one(self, tmp_path):
        data = _valid_data(overhead_buffer=1.5)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="overhead_buffer must be between 0 and 1"):
            load_client_config(path)

    def test_overhead_buffer_negative(self, tmp_path):
        data = _valid_data(overhead_buffer=-0.1)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="overhead_buffer must be between 0 and 1"):
            load_client_config(path)


class TestLanguage:
    def test_invalid_language(self, tmp_path):
        data = _valid_data(language="fr")
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="language must be 'es' or 'en'"):
            load_client_config(path)


class TestThresholds:
    def test_critical_must_be_greater_than_watch(self, tmp_path):
        data = _valid_data(
            backlog_threshold_watch=2.0,
            backlog_threshold_critical=1.0,
        )
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="backlog_threshold_critical must be greater than"):
            load_client_config(path)

    def test_critical_equal_to_watch_is_invalid(self, tmp_path):
        data = _valid_data(
            backlog_threshold_watch=1.5,
            backlog_threshold_critical=1.5,
        )
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="backlog_threshold_critical must be greater than"):
            load_client_config(path)


class TestInitialBacklog:
    def test_negative_initial_backlog_outbound(self, tmp_path):
        data = _valid_data(initial_backlog_outbound=-5)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="initial_backlog_outbound must be non-negative"):
            load_client_config(path)

    def test_negative_initial_backlog_inbound(self, tmp_path):
        data = _valid_data(initial_backlog_inbound=-1)
        path = _write_yaml(tmp_path, data)

        with pytest.raises(ConfigError, match="initial_backlog_inbound must be non-negative"):
            load_client_config(path)


class TestTargetBacklogRatio:
    def test_valid_ratio(self, tmp_path):
        data = _valid_data(target_backlog_ratio=0.5)
        path = _write_yaml(tmp_path, data)
        config = load_client_config(path)
        assert config.target_backlog_ratio == 0.5

    def test_ratio_above_one(self, tmp_path):
        data = _valid_data(target_backlog_ratio=1.5)
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="target_backlog_ratio must be between 0 and 1"):
            load_client_config(path)

    def test_ratio_negative(self, tmp_path):
        data = _valid_data(target_backlog_ratio=-0.1)
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="target_backlog_ratio must be between 0 and 1"):
            load_client_config(path)


class TestCurrentStaffing:
    def test_valid_staffing(self, tmp_path):
        data = _valid_data(current_staffing_outbound=10, current_staffing_inbound=5)
        path = _write_yaml(tmp_path, data)
        config = load_client_config(path)
        assert config.current_staffing_outbound == 10
        assert config.current_staffing_inbound == 5

    def test_negative_staffing_outbound(self, tmp_path):
        data = _valid_data(current_staffing_outbound=-1)
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="current_staffing_outbound must be non-negative"):
            load_client_config(path)

    def test_negative_staffing_inbound(self, tmp_path):
        data = _valid_data(current_staffing_inbound=-1)
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="current_staffing_inbound must be non-negative"):
            load_client_config(path)


class TestCostPerHour:
    def test_default_is_zero(self, tmp_path):
        path = _write_yaml(tmp_path, _valid_data())
        config = load_client_config(path)
        assert config.cost_per_hour == 0.0

    def test_negative_raises_error(self, tmp_path):
        data = _valid_data(cost_per_hour=-5.0)
        path = _write_yaml(tmp_path, data)
        with pytest.raises(ConfigError, match="cost_per_hour must be non-negative"):
            load_client_config(path)

    def test_custom_value_loads(self, tmp_path):
        data = _valid_data(cost_per_hour=12.50)
        path = _write_yaml(tmp_path, data)
        config = load_client_config(path)
        assert config.cost_per_hour == 12.50


class TestFileErrors:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(ConfigError, match="Config file not found"):
            load_client_config(tmp_path / "nonexistent.yaml")

    def test_non_mapping_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="must contain a YAML mapping"):
            load_client_config(p)
