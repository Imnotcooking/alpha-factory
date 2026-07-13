#!/usr/bin/env python3
"""Run the cumulative offline synthetic gate for Paper 1."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "notebooks"
    / "Phase_7_Research_Projects"
    / "07_01_daily_latent_regimes_cn_futures"
    / "config"
    / "smoke.yaml"
)

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.research.daily_regimes.smoke import main  # noqa: E402


def _arguments() -> list[str]:
    arguments = sys.argv[1:]
    if "--config" not in arguments:
        arguments = ["--config", str(DEFAULT_CONFIG), *arguments]
    return arguments


if __name__ == "__main__":
    raise SystemExit(main(_arguments()))
