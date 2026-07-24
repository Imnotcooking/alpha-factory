#!/usr/bin/env python3
"""CLI wrapper for package-owned IBKR adapter heartbeat checks."""

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

from oqp.ops.ibkr_heartbeat import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
