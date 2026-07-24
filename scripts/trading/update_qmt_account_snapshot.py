#!/usr/bin/env python3
"""Fetch QMT connector account state and write the unified account ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    account_snapshot_from_broker_snapshot,
    default_account_ledger_path,
    write_account_snapshot,
)
from oqp.brokers import BrokerConnectionStatus, get_broker_adapter, get_broker_profile_config  # noqa: E402
from oqp.config import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update the unified account ledger from the QMT connector.",
    )
    parser.add_argument(
        "--profile",
        default="qmt_paper_readonly",
        choices=("qmt_paper_readonly", "qmt_live_readonly"),
        help="QMT broker profile to snapshot.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date to write into account_nav, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="SQLite account ledger path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env_file)
    config = get_broker_profile_config(args.profile, settings=settings)
    broker = get_broker_adapter("qmt", settings=settings)
    health = broker.connect(config)
    if health.status != BrokerConnectionStatus.CONNECTED:
        broker.disconnect()
        print(
            json.dumps(
                {
                    "status": "fail",
                    "profile": args.profile,
                    "message": health.message or "QMT connector did not connect.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    try:
        broker_snapshot = broker.get_snapshot()
        account_snapshot = account_snapshot_from_broker_snapshot(
            broker_snapshot,
            environment=config.environment.value,
            profile=str(config.metadata.get("profile") or args.profile),
            broker_label="QMT",
            snapshot_date=args.snapshot_date,
        )
        result = write_account_snapshot(
            args.account_ledger_path,
            account_snapshot,
            snapshot_date=args.snapshot_date,
        )
    finally:
        broker.disconnect()

    print(json.dumps({"status": "updated", **result.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
