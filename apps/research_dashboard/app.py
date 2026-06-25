"""Compatibility entrypoint for the active research dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps._compat import run_legacy_streamlit_script


run_legacy_streamlit_script(
    "alpha_research_lab/ui_v2/app.py",
    legacy_root="alpha_research_lab/ui_v2",
)
