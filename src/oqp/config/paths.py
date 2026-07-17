"""Shared repository path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_repo_root(
    *,
    configured_root: str | Path | None = None,
    start: str | Path | None = None,
) -> Path:
    """Resolve the runtime repository without relying on package depth.

    Source checkouts can be discovered from their ``pyproject.toml`` and
    ``src/oqp`` markers. Installed deployments should set ``OQP_REPO_ROOT`` or
    launch from the repository working directory.
    """

    configured = configured_root or os.environ.get("OQP_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    starts = [Path(start).expanduser().resolve()] if start is not None else []
    starts.extend((Path.cwd().resolve(), Path(__file__).resolve()))
    seen: set[Path] = set()
    for initial in starts:
        candidate = initial if initial.is_dir() else initial.parent
        for root in (candidate, *candidate.parents):
            if root in seen:
                continue
            seen.add(root)
            if (root / "pyproject.toml").is_file() and (root / "src" / "oqp").is_dir():
                return root

    return Path(__file__).resolve().parents[3]


REPO_ROOT = resolve_repo_root()


__all__ = ["REPO_ROOT", "resolve_repo_root"]
