"""Load and validate PDP-SIM scenario configuration files."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError


class ConfigurationError(ValueError):
    """Raised when a scenario configuration is missing or invalid."""


def schema_path() -> Path:
    """Return the installed path to the canonical scenario schema."""
    return Path(str(files("schemadef").joinpath("config_schema.json")))


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and validate a scenario JSON file against the bundled schema."""
    config_path = Path(path)
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ConfigurationError("Invalid configuration at <root>: expected an object")
    config: dict[str, Any] = loaded

    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    errors = sorted(Draft7Validator(schema).iter_errors(config), key=lambda item: list(item.path))
    if errors:
        validation_error: ValidationError = errors[0]
        location = ".".join(str(part) for part in validation_error.absolute_path) or "<root>"
        raise ConfigurationError(f"Invalid configuration at {location}: {validation_error.message}")
    return config
