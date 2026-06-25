#!/usr/bin/env python3
"""Update the live portfolio position ledger from broker snapshots/exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    DEFAULT_BROKER_EXPORTS_DIR,
    DEFAULT_PORTFOLIO_EXPORTS_DIR,
    DEFAULT_PORTFOLIO_STATE_DIR,
    default_portfolio_ledger_path,
    run_portfolio_ingestion,
)
from oqp.accounts import default_account_ledger_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch live IBKR read-only positions, import broker CSV exports, "
            "and write the unified live_positions ledger."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=str(default_portfolio_ledger_path()),
        help="SQLite portfolio ledger path.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Snapshot date to write, in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(DEFAULT_BROKER_EXPORTS_DIR),
        help="Directory containing broker CSV exports.",
    )
    parser.add_argument(
        "--state-dir",
        default=str(DEFAULT_PORTFOLIO_STATE_DIR),
        help="Directory for portfolio state JSON outputs.",
    )
    parser.add_argument(
        "--backup-csv-dir",
        default=str(DEFAULT_PORTFOLIO_EXPORTS_DIR),
        help="Directory for optional unified portfolio CSV backups.",
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="SQLite account ledger path for unified live/paper account snapshots.",
    )
    parser.add_argument(
        "--no-legacy-raw-fallback",
        action="store_true",
        help="Do not read legacy Middle_Office/Portfolio/raw_data when raw-dir is empty.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_portfolio_ingestion(
        db_path=args.db_path,
        snapshot_date=args.snapshot_date,
        raw_dir=args.raw_dir,
        state_dir=args.state_dir,
        backup_csv_dir=args.backup_csv_dir,
        account_ledger_path=args.account_ledger_path,
        include_legacy_raw_fallback=not args.no_legacy_raw_fallback,
    )
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
