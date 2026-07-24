"""Bootstrap the frozen Paper 01 parity reference for shared-model tests."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
PAPER01_ROOT = (
    REPO_ROOT
    / "notebooks"
    / "Phase_7_Research_Projects"
    / "07_01_daily_latent_regimes_cn_futures"
)

paper_root = str(PAPER01_ROOT)
if paper_root not in sys.path:
    sys.path.insert(0, paper_root)
