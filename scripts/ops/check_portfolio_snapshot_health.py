#!/usr/bin/env python3
"""CLI wrapper for the package-owned portfolio snapshot health check."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.ops.portfolio_health import _discord_payload, main, print_checks, run_checks  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())
