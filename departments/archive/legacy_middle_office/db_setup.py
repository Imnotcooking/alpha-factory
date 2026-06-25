from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    default_portfolio_ledger_path,
    ensure_portfolio_ledger_schema,
)


DB_PATH = default_portfolio_ledger_path()


def init_db() -> None:
    ensure_portfolio_ledger_schema(DB_PATH)
    print("✅ Institutional SQLite Database Initialized.")


if __name__ == "__main__":
    init_db()
