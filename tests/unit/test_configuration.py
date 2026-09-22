from __future__ import annotations

import json
from pathlib import Path

import pytest

from configuration import ConfigurationError, load_config, schema_path


def test_bundled_schema_exists() -> None:
    assert schema_path().is_file()


def test_loads_repository_example() -> None:
    config = load_config(Path("tests/1_SBG_Timeseries/A_Velocity/input/config.json"))
    assert set(config) >= {"FLOW_PROPERTIES", "PROBE", "RECONSTRUCTION", "REPRODUCIBILITY"}


def test_reports_invalid_json(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid JSON"):
        load_config(config)


def test_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.json")


def test_reports_non_object_root(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="expected an object"):
        load_config(config)


def test_reports_schema_location(tmp_path: Path) -> None:
    source = Path("tests/1_SBG_Timeseries/A_Velocity/input/config.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    del payload["FLOW_PROPERTIES"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="FLOW_PROPERTIES"):
        load_config(config)
