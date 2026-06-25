"""Helpers for running legacy Streamlit scripts from the new app layout."""

from __future__ import annotations

import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_legacy_streamlit_script(
    relative_script: str,
    legacy_root: str | None = None,
) -> None:
    """Execute a legacy Streamlit script while preserving its path assumptions."""

    script_path = (REPO_ROOT / relative_script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy Streamlit script not found: {script_path}")

    legacy_root_path = (REPO_ROOT / legacy_root).resolve() if legacy_root else script_path.parent
    original_sys_path = list(sys.path)
    for path in (
        str(script_path.parent),
        str(legacy_root_path),
        str(REPO_ROOT / "src"),
        str(REPO_ROOT),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        with _temporary_cwd(legacy_root_path):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.path[:] = original_sys_path
