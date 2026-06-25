"""Compatibility wrapper for the unified portfolio ingestion job.

The canonical implementation now lives in ``src/oqp/portfolio/ingestion_job.py``.
This file remains temporarily so older Middle Office commands keep working while
the root ``Middle_Office`` folder is phased out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio.ingestion_job import (  # noqa: E402,F401
    DEFAULT_BROKER_EXPORTS_DIR,
    DEFAULT_PORTFOLIO_EXPORTS_DIR,
    DEFAULT_PORTFOLIO_STATE_DIR,
    fetch_live_ibkr_portfolio,
    fetch_polygon_greeks,
    futu_to_occ,
    process_futu_csv,
    process_t212_csv,
    run_portfolio_ingestion,
    save_ibkr_metrics,
)


if __name__ == "__main__":
    result = run_portfolio_ingestion()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
