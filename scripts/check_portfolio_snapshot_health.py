#!/usr/bin/env python3
"""Check whether the portfolio snapshot/NAV ledger is fresh enough."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import DEFAULT_IBKR_METRICS_PATH, default_portfolio_ledger_path  # noqa: E402


DEFAULT_METRICS_PATH = DEFAULT_IBKR_METRICS_PATH


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: str
    detail: str

    def failed(self) -> bool:
        return self.status == "fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the portfolio ledger has a fresh NAV snapshot.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_portfolio_ledger_path()),
        help="SQLite ledger path containing live_positions and historical_nav.",
    )
    parser.add_argument(
        "--ibkr-metrics-path",
        default=str(DEFAULT_METRICS_PATH),
        help="IBKR metrics JSON path produced by the broker ETL.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=36.0,
        help="Maximum acceptable age for the latest historical_nav date.",
    )
    parser.add_argument(
        "--expect-date",
        default=None,
        help="Require the latest NAV date to equal this YYYY-MM-DD date.",
    )
    parser.add_argument(
        "--status-path",
        default=None,
        help="Optional JSON status output path.",
    )
    parser.add_argument(
        "--webhook-url",
        default=(
            os.getenv("OQP_DISCORD_WEBHOOK_URL")
            or os.getenv("OQP_HEALTH_WEBHOOK_URL")
        ),
        help=(
            "Optional Discord webhook URL. Also read from "
            "OQP_DISCORD_WEBHOOK_URL or OQP_HEALTH_WEBHOOK_URL."
        ),
    )
    parser.add_argument(
        "--notify-always",
        action="store_true",
        help="Post webhook status even when all checks pass.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = run_checks(
        db_path=Path(args.db_path),
        ibkr_metrics_path=Path(args.ibkr_metrics_path),
        max_age_hours=args.max_age_hours,
        expect_date=args.expect_date,
    )
    failed = any(check.failed() for check in checks)
    payload = {
        "status": "fail" if failed else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": [asdict(check) for check in checks],
    }

    if args.status_path:
        status_path = Path(args.status_path)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if args.webhook_url and (failed or args.notify_always):
        _post_webhook(args.webhook_url, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_checks(checks)

    return 1 if failed else 0


def run_checks(
    *,
    db_path: Path,
    ibkr_metrics_path: Path,
    max_age_hours: float,
    expect_date: str | None = None,
) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    checks.append(
        HealthCheck(
            "portfolio ledger",
            "pass" if db_path.exists() else "fail",
            f"Found {db_path}." if db_path.exists() else f"Missing {db_path}.",
        )
    )
    if not db_path.exists():
        checks.append(_file_freshness_check("ibkr metrics", ibkr_metrics_path, max_age_hours))
        return checks

    try:
        with sqlite3.connect(db_path) as conn:
            tables = _tables(conn)
            checks.append(_table_check(tables, "historical_nav"))
            checks.append(_table_check(tables, "live_positions"))

            if "historical_nav" in tables:
                checks.extend(
                    _historical_nav_checks(
                        conn,
                        max_age_hours=max_age_hours,
                        expect_date=expect_date,
                    )
                )
            if "live_positions" in tables:
                checks.append(_live_positions_check(conn))
    except sqlite3.Error as exc:
        checks.append(HealthCheck("sqlite read", "fail", str(exc)))

    checks.append(_file_freshness_check("ibkr metrics", ibkr_metrics_path, max_age_hours))
    return checks


def print_checks(checks: list[HealthCheck]) -> None:
    width = max(len(check.name) for check in checks) if checks else 0
    for check in checks:
        print(f"{check.status.upper():5}  {check.name:<{width}}  {check.detail}")


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_check(tables: set[str], table: str) -> HealthCheck:
    return HealthCheck(
        table,
        "pass" if table in tables else "fail",
        "Table exists." if table in tables else "Table is missing.",
    )


def _historical_nav_checks(
    conn: sqlite3.Connection,
    *,
    max_age_hours: float,
    expect_date: str | None,
) -> list[HealthCheck]:
    latest = conn.execute(
        """
        SELECT date, total_net_worth, total_cash, portfolio_beta, daily_pnl
        FROM historical_nav
        ORDER BY date DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return [
            HealthCheck(
                "latest NAV",
                "fail",
                "historical_nav has no rows.",
            )
        ]

    latest_date = str(latest[0])
    total_net_worth = _float(latest[1])
    checks = [
        HealthCheck(
            "latest NAV",
            "pass",
            (
                f"date={latest_date} total_net_worth={total_net_worth:,.2f} "
                f"cash={_float(latest[2]):,.2f} beta={_float(latest[3]):,.4f}"
            ),
        ),
        HealthCheck(
            "NAV positive",
            "pass" if total_net_worth > 0 else "fail",
            (
                "Latest NAV is positive."
                if total_net_worth > 0
                else f"Latest NAV is not positive: {total_net_worth:,.2f}."
            ),
        ),
    ]

    age_hours = _date_age_hours(latest_date)
    if age_hours is None:
        checks.append(
            HealthCheck(
                "NAV freshness",
                "fail",
                f"Could not parse latest NAV date: {latest_date}.",
            )
        )
    else:
        checks.append(
            HealthCheck(
                "NAV freshness",
                "pass" if age_hours <= max_age_hours else "fail",
                (
                    f"Latest NAV is {age_hours:.1f} hours old "
                    f"(limit {max_age_hours:.1f})."
                ),
            )
        )

    if expect_date:
        checks.append(
            HealthCheck(
                "expected NAV date",
                "pass" if latest_date == expect_date else "fail",
                (
                    f"Latest NAV date matches {expect_date}."
                    if latest_date == expect_date
                    else f"Latest NAV date is {latest_date}, expected {expect_date}."
                ),
            )
        )

    return checks


def _live_positions_check(conn: sqlite3.Connection) -> HealthCheck:
    latest = conn.execute(
        """
        SELECT date, COUNT(*) AS rows
        FROM live_positions
        GROUP BY date
        ORDER BY date DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return HealthCheck(
            "latest positions",
            "warn",
            "live_positions has no rows. This can be normal for cash-only accounts.",
        )
    return HealthCheck(
        "latest positions",
        "pass",
        f"date={latest[0]} rows={latest[1]}.",
    )


def _file_freshness_check(name: str, path: Path, max_age_hours: float) -> HealthCheck:
    if not path.exists():
        return HealthCheck(name, "warn", f"Missing {path}.")
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - modified).total_seconds() / 3600
    return HealthCheck(
        name,
        "pass" if age_hours <= max_age_hours else "warn",
        f"Modified {age_hours:.1f} hours ago at {modified.isoformat()}.",
    )


def _date_age_hours(value: str) -> float | None:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    then = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - then).total_seconds() / 3600


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    body = json.dumps(_discord_payload(payload), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-Portfolio-Health/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   webhook  Could not post health status: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARN   webhook  Could not post health status: {exc}", file=sys.stderr)


def _discord_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "unknown")).upper()
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        checks = []

    failed = [check for check in checks if check.get("status") == "fail"]
    warnings = [check for check in checks if check.get("status") == "warn"]
    latest_nav = _find_check(checks, "latest NAV")
    freshness = _find_check(checks, "NAV freshness")
    latest_positions = _find_check(checks, "latest positions")

    fields: list[dict[str, Any]] = []
    if latest_nav:
        fields.append(_discord_field("Latest NAV", latest_nav.get("detail", "")))
    if freshness:
        fields.append(_discord_field("Freshness", freshness.get("detail", "")))
    if latest_positions:
        fields.append(_discord_field("Positions", latest_positions.get("detail", "")))

    for check in failed[:5]:
        fields.append(
            _discord_field(
                f"FAIL: {check.get('name', 'check')}",
                str(check.get("detail", "")),
            )
        )
    if not failed:
        for check in warnings[:3]:
            fields.append(
                _discord_field(
                    f"WARN: {check.get('name', 'check')}",
                    str(check.get("detail", "")),
                )
            )

    return {
        "username": "OQP Portfolio Health",
        "content": f"OQP portfolio snapshot health: {status}",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Portfolio Snapshot Health",
                "description": f"Status: {status}",
                "color": 0x2ECC71 if status == "PASS" else 0xE74C3C,
                "timestamp": payload.get("checked_at"),
                "fields": fields[:10],
            }
        ],
    }


def _find_check(checks: list[Any], name: str) -> dict[str, Any] | None:
    for check in checks:
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return None


def _discord_field(name: str, value: str) -> dict[str, Any]:
    text = value.strip() or "No detail."
    return {
        "name": name[:256],
        "value": text[:1024],
        "inline": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
