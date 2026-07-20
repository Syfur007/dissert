from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def _resolve_include_path(current_file: Path, include: str | Path) -> Path:
    include_path = Path(include).expanduser()
    if not include_path.is_absolute():
        include_path = current_file.parent / include_path
    return include_path.resolve()


def _load_config_file(config_path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Circular config composition detected: {cycle}")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text())
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration file must contain a mapping at the top level: {config_path}")

    compose_entries = raw_config.pop("compose", [])
    if compose_entries is None:
        compose_entries = []
    if not isinstance(compose_entries, list):
        raise TypeError(f"'compose' must be a list in {config_path}")

    composed: dict[str, Any] = {}
    next_stack = stack + (config_path,)

    for include in compose_entries:
        if not isinstance(include, (str, Path)):
            raise TypeError(
                f"Each compose entry must be a path string in {config_path}; got {type(include).__name__}"
            )
        include_path = _resolve_include_path(config_path, include)
        included_config = _load_config_file(include_path, next_stack)
        composed = _merge_dicts(composed, included_config)

    return _merge_dicts(composed, raw_config)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and recursively compose any referenced fragments."""
    return _load_config_file(Path(config_path).expanduser().resolve(), ())