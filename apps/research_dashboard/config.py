"""Compatibility wrapper for shared research dashboard UI config."""

import os
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
APPS_DIR = APP_DIR.parent
REPO_ROOT = APPS_DIR.parent
UI_DIR = APP_DIR
BASE_DIR = REPO_ROOT
SRC_DIR = REPO_ROOT / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from oqp.ui.research_dashboard_config import *  # noqa: F401,F403,E402

RUNTIME_ROOT = REPO_ROOT / "runtime"
ALPHA_RUNTIME_DATA_ROOT = RUNTIME_ROOT / "data" / "alpha_lab"
ALPHA_RUNTIME_ARTIFACT_ROOT = RUNTIME_ROOT / "artifacts" / "research" / "alpha_lab"
RESEARCH_ARTIFACT_ROOT = Path(
    os.environ.get(
        "ALPHA_RESEARCH_ARTIFACT_ROOT",
        os.environ.get("ALPHA_RUNTIME_ARTIFACT_ROOT", ALPHA_RUNTIME_ARTIFACT_ROOT),
    )
)

DB_PATH = str(
    Path(
        os.environ.get(
            "ALPHA_RESEARCH_DB_PATH",
            RUNTIME_ROOT / "db" / "research" / "alpha_lab" / "research_memory.db",
        )
    )
)

# Compatibility alias for older dashboard modules. This is the research artifact
# root, not runtime/logs; backtest returns/trades/cache files live below it.
LOGS_DIR = str(RESEARCH_ARTIFACT_ROOT)
