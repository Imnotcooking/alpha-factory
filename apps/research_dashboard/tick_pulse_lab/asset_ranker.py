"""Compatibility wrapper for promoted tick-pulse asset ranking helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from oqp.research.tick_pulse.asset_ranker import *  # noqa: F401,F403
