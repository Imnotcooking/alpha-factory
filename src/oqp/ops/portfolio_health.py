#!/usr/bin/env python3
"""Check whether the portfolio snapshot/NAV ledger is fresh enough."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


from oqp.accounts import UNIFIED_LIVE_PROFILE, default_account_ledger_path
from oqp.ops.notifications import discord_field, post_json_webhook
from oqp.portfolio import DEFAULT_IBKR_METRICS_PATH, default_portfolio_ledger_path


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
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="Unified account ledger path containing account_nav/account_positions.",
    )
    parser.add_argument(
        "--account-profile",
        default=UNIFIED_LIVE_PROFILE,
        help="Live account profile to validate inside the account ledger.",
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
        account_ledger_path=Path(args.account_ledger_path),
        account_profile=args.account_profile,
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
    account_ledger_path: Path | None = None,
    account_profile: str = UNIFIED_LIVE_PROFILE,
    ibkr_metrics_path: Path = DEFAULT_METRICS_PATH,
    max_age_hours: float = 36.0,
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

    if account_ledger_path is not None:
        checks.extend(
            _account_ledger_checks(
                account_ledger_path,
                account_profile=account_profile,
                max_age_hours=max_age_hours,
                expect_date=expect_date,
            )
        )
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


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


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


def _account_ledger_checks(
    account_ledger_path: Path,
    *,
    account_profile: str,
    max_age_hours: float,
    expect_date: str | None,
) -> list[HealthCheck]:
    checks = [
        HealthCheck(
            "account ledger",
            "pass" if account_ledger_path.exists() else "warn",
            (
                f"Found {account_ledger_path}."
                if account_ledger_path.exists()
                else f"Missing {account_ledger_path}."
            ),
        )
    ]
    if not account_ledger_path.exists():
        return checks

    try:
        with sqlite3.connect(account_ledger_path) as conn:
            tables = _tables(conn)
            checks.append(_table_check(tables, "account_nav"))
            checks.append(_table_check(tables, "account_positions"))
            if "account_nav" in tables:
                checks.extend(
                    _account_nav_checks(
                        conn,
                        account_profile=account_profile,
                        max_age_hours=max_age_hours,
                        expect_date=expect_date,
                    )
                )
            if "account_positions" in tables:
                checks.append(
                    _account_positions_check(
                        conn,
                        account_profile=account_profile,
                    )
                )
    except sqlite3.Error as exc:
        checks.append(HealthCheck("account sqlite read", "fail", str(exc)))
    return checks


def _account_nav_checks(
    conn: sqlite3.Connection,
    *,
    account_profile: str,
    max_age_hours: float,
    expect_date: str | None,
) -> list[HealthCheck]:
    currency_expr = "currency" if _table_has_column(conn, "account_nav", "currency") else "'USD' AS currency"
    latest = conn.execute(
        f"""
        SELECT date, {currency_expr}, net_liquidation, cash, daily_pnl, position_count, as_of
        FROM account_nav
        WHERE environment = 'live' AND profile = ?
        ORDER BY as_of DESC
        LIMIT 1
        """,
        (account_profile,),
    ).fetchone()
    if latest is None:
        return [
            HealthCheck(
                "unified live NAV",
                "warn",
                f"No live account_nav row found for profile={account_profile}.",
            )
        ]

    latest_date = str(latest[0])
    currency = str(latest[1] or "USD").upper()
    nav = _float(latest[2])
    as_of = str(latest[6])
    checks = [
        HealthCheck(
            "unified live NAV",
            "pass",
            (
                f"date={latest_date} currency={currency} nav={nav:,.2f} cash={_float(latest[3]):,.2f} "
                f"daily_pnl={_float(latest[4]):,.2f} positions={int(_float(latest[5]))} "
                f"as_of={as_of}"
            ),
        ),
        HealthCheck(
            "unified NAV positive",
            "pass" if nav > 0 else "fail",
            "Unified live NAV is positive." if nav > 0 else f"Unified live NAV is {nav:,.2f}.",
        ),
    ]

    age_hours = _datetime_age_hours(as_of)
    checks.append(
        HealthCheck(
            "unified NAV freshness",
            "pass" if age_hours is not None and age_hours <= max_age_hours else "fail",
            (
                f"Unified live NAV is {age_hours:.1f} hours old (limit {max_age_hours:.1f})."
                if age_hours is not None
                else f"Could not parse unified live as_of: {as_of}."
            ),
        )
    )
    if expect_date:
        checks.append(
            HealthCheck(
                "expected account NAV date",
                "pass" if latest_date == expect_date else "fail",
                (
                    f"Unified live NAV date matches {expect_date}."
                    if latest_date == expect_date
                    else f"Unified live NAV date is {latest_date}, expected {expect_date}."
                ),
            )
        )
    return checks


def _account_positions_check(
    conn: sqlite3.Connection,
    *,
    account_profile: str,
) -> HealthCheck:
    latest = conn.execute(
        """
        SELECT s.snapshot_date, s.snapshot_id, COUNT(p.symbol)
        FROM account_snapshots s
        LEFT JOIN account_positions p ON p.snapshot_id = s.snapshot_id
        WHERE s.environment = 'live' AND s.profile = ?
        GROUP BY s.snapshot_date, s.snapshot_id, s.as_of
        ORDER BY s.as_of DESC
        LIMIT 1
        """,
        (account_profile,),
    ).fetchone()
    if latest is None:
        return HealthCheck(
            "unified live positions",
            "warn",
            f"No live account_positions snapshot found for profile={account_profile}.",
        )
    return HealthCheck(
        "unified live positions",
        "pass",
        f"date={latest[0]} snapshot_id={latest[1]} rows={latest[2]}.",
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


def _datetime_age_hours(value: str) -> float | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    post_json_webhook(
        url,
        _discord_payload(payload),
        user_agent="OQP-Portfolio-Health/1.0",
        label="webhook  Could not post health status:",
    )


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
    account_nav = _find_check(checks, "unified live NAV")
    account_freshness = _find_check(checks, "unified NAV freshness")
    account_positions = _find_check(checks, "unified live positions")

    fields: list[dict[str, Any]] = []
    if account_nav:
        fields.append(_discord_field("Unified Live NAV", account_nav.get("detail", "")))
    if account_freshness:
        fields.append(_discord_field("Unified Freshness", account_freshness.get("detail", "")))
    if account_positions:
        fields.append(_discord_field("Unified Positions", account_positions.get("detail", "")))
    if latest_nav:
        fields.append(_discord_field("Legacy NAV", latest_nav.get("detail", "")))
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
    return discord_field(name, value)


if __name__ == "__main__":
    raise SystemExit(main())
