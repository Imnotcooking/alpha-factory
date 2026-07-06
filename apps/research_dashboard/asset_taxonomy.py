"""Compatibility wrapper for promoted OQP asset taxonomy helpers."""

from __future__ import annotations

import sys
from pathlib import Path


UI_DIR = Path(__file__).resolve().parent
SRC_DIR = UI_DIR.parents[1] / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oqp.data.asset_taxonomy import *  # noqa: F401,F403,E402
