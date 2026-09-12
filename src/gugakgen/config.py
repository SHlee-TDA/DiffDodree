"""Small configuration helpers shared by Python workflows and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start)
    current = current.expanduser().resolve()
    candidates = (current, *current.parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"Could not locate pyproject.toml from {current}")


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"YAML root must be a mapping: {source}")
    return dict(value)


def project_path(project_root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section {name!r} must be a mapping")
    return dict(value)
