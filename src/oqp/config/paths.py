"""Shared repository path helpers."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_MIDDLE_OFFICE_ROOT_PATH = REPO_ROOT / "Middle_Office"
ARCHIVED_LEGACY_MIDDLE_OFFICE_ROOT_PATH = (
    REPO_ROOT / "departments" / "archive" / "legacy_middle_office"
)


def _path_from_env(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def legacy_middle_office_root() -> Path:
    """Return the best available legacy Middle Office root."""

    configured = os.getenv("OQP_LEGACY_MIDDLE_OFFICE_ROOT")
    if configured:
        return _path_from_env(configured)
    if LEGACY_MIDDLE_OFFICE_ROOT_PATH.exists():
        return LEGACY_MIDDLE_OFFICE_ROOT_PATH
    return ARCHIVED_LEGACY_MIDDLE_OFFICE_ROOT_PATH
