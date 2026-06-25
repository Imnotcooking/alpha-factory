#!/usr/bin/env python3
"""Fetch IBKR paper account state and write the paper trading ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.brokers import (  # noqa: E402
    BrokerConnectionStatus,
    fetch_ibkr_readonly_portfolio_snapshot,
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import load_settings  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    default_paper_trading_ledger_path,
    write_paper_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the read-only IBKR paper trading snapshot ledger.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_paper_trading_ledger_path()),
        help="SQLite paper trading ledger path.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date to write into paper_nav, in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env_file)
    config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
    broker = get_broker_adapter("ibkr", settings=settings)
    snapshot = fetch_ibkr_readonly_portfolio_snapshot(config, adapter=broker)

    if snapshot.health.status != BrokerConnectionStatus.CONNECTED or snapshot.error:
        message = snapshot.error or snapshot.health.message or "Paper adapter did not connect."
        print(json.dumps({"status": "fail", "message": message}, indent=2))
        return 1

    result = write_paper_snapshot(
        args.db_path,
        snapshot,
        snapshot_date=args.snapshot_date,
    )
    print(json.dumps({"status": "updated", **result.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
