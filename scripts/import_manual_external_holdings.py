#!/usr/bin/env python3
"""Import manual external holdings into the unified account ledger."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    default_account_ledger_path,
    load_manual_external_positions,
    upsert_manual_external_positions,
)


DEFAULT_INPUT = REPO_ROOT / "runtime" / "state" / "portfolio" / "manual_external_holdings.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSON file with manual external positions.")
    parser.add_argument("--db", type=Path, default=default_account_ledger_path(), help="Account ledger SQLite path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print without writing.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("positions", payload)
    if not isinstance(rows, list):
        raise SystemExit("Input JSON must be a list or an object with a positions list.")

    if args.dry_run:
        print(f"validated_rows={len(rows)}")
        for row in rows:
            print(f"- {row.get('position_id')}: {row.get('symbol')} {row.get('quantity')}")
        return 0

    written = upsert_manual_external_positions(args.db, rows)
    loaded = load_manual_external_positions(args.db, environment="live")
    print(f"db={args.db}")
    print(f"upserted={written}")
    print(f"active_live_rows={len(loaded)}")
    if not loaded.empty:
        print(loaded[["position_id", "symbol", "asset_class", "quantity", "currency", "local_market_value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
