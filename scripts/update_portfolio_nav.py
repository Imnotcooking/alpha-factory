#!/usr/bin/env python3
"""Update the shared portfolio NAV ledger from the command line."""

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
    DEFAULT_IBKR_METRICS_PATH,
    DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH,
    default_portfolio_ledger_path,
    update_portfolio_nav,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch market prices, value the latest portfolio snapshot, and write daily NAV.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_portfolio_ledger_path()),
        help="SQLite ledger path containing live_positions and historical_nav.",
    )
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="NAV date to write, in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--period",
        default="6mo",
        help="Yahoo Finance lookback window used for valuation and beta.",
    )
    parser.add_argument(
        "--benchmark",
        default="QQQ",
        help="Benchmark ticker used for correlation and beta.",
    )
    parser.add_argument(
        "--defaults-path",
        default=str(DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH),
        help="Non-secret dashboard defaults JSON with manual cash/assets.",
    )
    parser.add_argument(
        "--ibkr-metrics-path",
        default=str(DEFAULT_IBKR_METRICS_PATH),
        help="IBKR metrics JSON produced by the broker ETL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate NAV without writing historical_nav.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = update_portfolio_nav(
        db_path=args.db_path,
        snapshot_date=args.snapshot_date,
        period=args.period,
        benchmark=args.benchmark,
        defaults_path=args.defaults_path,
        ibkr_metrics_path=args.ibkr_metrics_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
