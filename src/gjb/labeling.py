"""Validation helpers for label-dependent experiment inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


EMPTY_VALUES = {"", "na", "n/a", "none", "null", "unknown"}
TRUE_VALUES = {"1", "true", "t", "yes", "y", "success", "succeeded", "passed", "pass"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "fail", "failed", "unsuccessful", "blocked"}


def parse_optional_bool(value: Any, field_name: str = "value") -> bool | None:
    """Parse common boolean label encodings while preserving blank values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    text = str(value).strip().lower()
    if text in EMPTY_VALUES:
        return None
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for {field_name}: {value!r}")


def parse_required_bool(value: Any, field_name: str = "value") -> bool:
    parsed = parse_optional_bool(value, field_name)
    if parsed is None:
        raise ValueError(f"Missing required boolean value for {field_name}")
    return parsed


def parse_turns_json(value: str | None, field_name: str = "turns_json") -> list[str] | None:
    """Parse a JSON turn list without logging prompt content.

    Accepted shapes:
    - ["turn 1", "turn 2"]
    - [{"role": "user", "content": "turn 1"}, ...]
    """
    if value is None or value.strip().lower() in EMPTY_VALUES:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{field_name} must be a non-empty JSON list")

    turns: list[str] = []
    for index, item in enumerate(parsed, start=1):
        if isinstance(item, str):
            content = item.strip()
        elif isinstance(item, dict):
            content = str(item.get("content", "")).strip()
        else:
            raise ValueError(f"{field_name}[{index}] must be a string or an object with content")
        if content:
            turns.append(content)

    if not turns:
        raise ValueError(f"{field_name} did not contain any non-empty turn content")
    return turns


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_prompt_hash(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("prompt_sha256") or sha256_text(row.get("prompt", "")))


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def require_unique(values: Iterable[str], field_name: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Duplicate {field_name} values: {sorted(duplicates)}")
