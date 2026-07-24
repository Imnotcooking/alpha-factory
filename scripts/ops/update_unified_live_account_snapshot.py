#!/usr/bin/env python3
"""Materialize the unified live account snapshot in the account ledger."""

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
    DEFAULT_LIVE_BROKER_PROFILE,
    UNIFIED_LIVE_PROFILE,
    default_account_ledger_path,
    materialize_unified_live_account_snapshot,
    sync_manual_external_positions_from_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync manual external holdings from JSON, then write the combined "
            "IBKR-plus-manual live account snapshot."
        ),
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="SQLite account ledger path.",
    )
    parser.add_argument(
        "--broker-profile",
        default=DEFAULT_LIVE_BROKER_PROFILE,
        help="Live broker profile to layer manual holdings onto.",
    )
    parser.add_argument(
        "--unified-profile",
        default=UNIFIED_LIVE_PROFILE,
        help="Unified live profile name to write.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.account_ledger_path)
    manual_rows = sync_manual_external_positions_from_json(path)
    result = materialize_unified_live_account_snapshot(
        path,
        broker_profile=args.broker_profile,
        unified_profile=args.unified_profile,
    )
    payload = result.to_dict()
    payload["manual_sync_rows"] = manual_rows
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
