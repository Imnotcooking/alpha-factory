"""Credential lookup helpers for phased migration.

Environment variables and ``.env`` values always win. Legacy JSON files are
read as a transition path only so existing local dashboards keep working while
the repo moves toward centralized runtime config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


JsonCredentialSource = tuple[Path, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class CredentialLookup:
    value: str | None
    source: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.value)


def clean_secret(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().strip('"').strip("'")
    return cleaned or None


def load_json_secret(path: Path, keys: Iterable[str]) -> CredentialLookup:
    if not path.exists():
        return CredentialLookup(value=None)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CredentialLookup(value=None)

    if not isinstance(payload, dict):
        return CredentialLookup(value=None)

    for key in keys:
        value = clean_secret(payload.get(key))
        if value:
            return CredentialLookup(value=value, source=f"{path}:{key}")
    return CredentialLookup(value=None)


def save_json_secret(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            payload = {}

    payload[key] = value
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_credential(
    env_names: Iterable[str],
    env_values: dict[str, str] | None = None,
    json_sources: Iterable[JsonCredentialSource] = (),
) -> CredentialLookup:
    local_env = env_values or {}

    for name in env_names:
        value = clean_secret(os.getenv(name) or local_env.get(name))
        if value:
            return CredentialLookup(value=value, source=f"env:{name}")

    for path, keys in json_sources:
        lookup = load_json_secret(path, keys)
        if lookup.found:
            return lookup

    return CredentialLookup(value=None)


def mask_secret(value: str | None, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}...{value[-visible:]}"
