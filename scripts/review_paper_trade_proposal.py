#!/usr/bin/env python3
"""CLI wrapper for package-owned paper trade proposal review."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.paper_trading.proposal_review_command import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
