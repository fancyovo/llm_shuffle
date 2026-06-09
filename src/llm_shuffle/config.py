from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and key in result
            and isinstance(result[key], dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = load_yaml(path)
    include = cfg.pop("include", None)
    merged: dict[str, Any] = {}
    if include:
        for include_path in include.values():
            include_cfg = load_config(include_path)
            merged = deep_update(merged, include_cfg)
    return deep_update(merged, cfg)
