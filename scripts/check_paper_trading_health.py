#!/usr/bin/env python3
"""Check paper trading ledger freshness and optionally send a Discord report."""

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

from oqp.paper_trading import default_paper_trading_ledger_path  # noqa: E402


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: str
    detail: str

    def failed(self) -> bool:
        return self.status == "fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the paper trading ledger has a fresh snapshot.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_paper_trading_ledger_path()),
        help="SQLite paper trading ledger path.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=36.0,
        help="Maximum acceptable age for the latest paper NAV snapshot.",
    )
    parser.add_argument(
        "--status-path",
        default=None,
        help="Optional JSON status output path.",
    )
    parser.add_argument(
        "--webhook-url",
        default=(
            os.getenv("OQP_PAPER_DISCORD_WEBHOOK_URL")
            or os.getenv("OQP_DISCORD_WEBHOOK_URL")
            or os.getenv("OQP_HEALTH_WEBHOOK_URL")
        ),
        help="Optional Discord webhook URL.",
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
    checks, summary = run_checks(
        db_path=Path(args.db_path),
        max_age_hours=args.max_age_hours,
    )
    failed = any(check.failed() for check in checks)
    payload = {
        "status": "fail" if failed else "pass",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
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
    max_age_hours: float,
) -> tuple[list[HealthCheck], dict[str, Any]]:
    checks: list[HealthCheck] = []
    summary: dict[str, Any] = {}
    checks.append(
        HealthCheck(
            "paper ledger",
            "pass" if db_path.exists() else "fail",
            f"Found {db_path}." if db_path.exists() else f"Missing {db_path}.",
        )
    )
    if not db_path.exists():
        return checks, summary

    try:
        with sqlite3.connect(db_path) as conn:
            tables = _tables(conn)
            for table in (
                "paper_account_snapshots",
                "paper_positions",
                "paper_nav",
                "paper_orders",
                "paper_fills",
            ):
                checks.append(_table_check(tables, table))

            if "paper_nav" in tables:
                nav_checks, nav_summary = _paper_nav_checks(
                    conn,
                    max_age_hours=max_age_hours,
                )
                checks.extend(nav_checks)
                summary.update(nav_summary)

            if "paper_positions" in tables:
                checks.append(_paper_positions_check(conn, summary))

            if "paper_orders" in tables and "paper_fills" in tables:
                checks.append(_paper_activity_check(conn, summary))
    except sqlite3.Error as exc:
        checks.append(HealthCheck("sqlite read", "fail", str(exc)))

    return checks, summary


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


def _paper_nav_checks(
    conn: sqlite3.Connection,
    *,
    max_age_hours: float,
) -> tuple[list[HealthCheck], dict[str, Any]]:
    latest = conn.execute(
        """
        SELECT date, account_id, as_of, net_liquidation, cash, daily_pnl,
               position_count, snapshot_id
        FROM paper_nav
        ORDER BY as_of DESC
        LIMIT 1
        """
    ).fetchone()
    if latest is None:
        return [HealthCheck("latest paper NAV", "fail", "paper_nav has no rows.")], {}

    summary = {
        "date": latest[0],
        "account_id": _redact_account(latest[1]),
        "as_of": latest[2],
        "net_liquidation": _float(latest[3]),
        "cash": _float(latest[4]),
        "daily_pnl": _float(latest[5]),
        "position_count": int(latest[6] or 0),
        "snapshot_id": latest[7],
    }
    nav = float(summary["net_liquidation"])
    checks = [
        HealthCheck(
            "latest paper NAV",
            "pass",
            (
                f"date={summary['date']} nav={nav:,.2f} "
                f"cash={float(summary['cash']):,.2f} "
                f"daily_pnl={float(summary['daily_pnl']):,.2f}"
            ),
        ),
        HealthCheck(
            "paper NAV positive",
            "pass" if nav > 0 else "fail",
            "Latest paper NAV is positive." if nav > 0 else f"Latest paper NAV is {nav}.",
        ),
    ]
    age_hours = _datetime_age_hours(str(summary["as_of"]))
    checks.append(
        HealthCheck(
            "paper freshness",
            "pass" if age_hours is not None and age_hours <= max_age_hours else "fail",
            (
                f"Latest paper snapshot is {age_hours:.1f} hours old "
                f"(limit {max_age_hours:.1f})."
                if age_hours is not None
                else f"Could not parse as_of: {summary['as_of']}."
            ),
        )
    )
    return checks, summary


def _paper_positions_check(
    conn: sqlite3.Connection,
    summary: dict[str, Any],
) -> HealthCheck:
    snapshot_id = summary.get("snapshot_id")
    if not snapshot_id:
        return HealthCheck("paper positions", "warn", "No latest snapshot id.")
    count = conn.execute(
        "SELECT COUNT(*) FROM paper_positions WHERE snapshot_id = ?",
        (snapshot_id,),
    ).fetchone()[0]
    summary["position_rows"] = int(count)
    return HealthCheck(
        "paper positions",
        "pass",
        f"snapshot_id={snapshot_id} rows={count}.",
    )


def _paper_activity_check(
    conn: sqlite3.Connection,
    summary: dict[str, Any],
) -> HealthCheck:
    today = date.today().isoformat()
    orders = conn.execute(
        "SELECT COUNT(*) FROM paper_orders WHERE created_at >= ?",
        (today,),
    ).fetchone()[0]
    fills = conn.execute(
        "SELECT COUNT(*) FROM paper_fills WHERE executed_at >= ?",
        (today,),
    ).fetchone()[0]
    summary["orders_today"] = int(orders)
    summary["fills_today"] = int(fills)
    return HealthCheck(
        "paper activity",
        "pass",
        f"orders_today={orders} fills_today={fills}.",
    )


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    body = json.dumps(_discord_payload(payload), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-Paper-Trading-Health/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   webhook  Could not post paper status: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARN   webhook  Could not post paper status: {exc}", file=sys.stderr)


def _discord_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "unknown")).upper()
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        checks = []

    failed = [check for check in checks if check.get("status") == "fail"]
    fields = [
        _discord_field("Paper Account", str(summary.get("account_id", "n/a"))),
        _discord_field("Paper NAV", _money(summary.get("net_liquidation"))),
        _discord_field("Daily P&L", _money(summary.get("daily_pnl"))),
        _discord_field("Cash", _money(summary.get("cash"))),
        _discord_field("Positions", str(summary.get("position_rows", summary.get("position_count", "n/a")))),
        _discord_field(
            "Activity",
            f"orders_today={summary.get('orders_today', 0)} fills_today={summary.get('fills_today', 0)}",
        ),
    ]
    for check in failed[:4]:
        fields.append(
            _discord_field(
                f"FAIL: {check.get('name', 'check')}",
                str(check.get("detail", "")),
            )
        )

    return {
        "username": "OQP Paper Trading",
        "content": f"OQP paper trading report: {status}",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Paper Trading Daily Report",
                "description": f"Status: {status}",
                "color": 0x3498DB if status == "PASS" else 0xE74C3C,
                "timestamp": payload.get("checked_at"),
                "fields": fields[:10],
            }
        ],
    }


def _discord_field(name: str, value: str) -> dict[str, Any]:
    text = value.strip() or "No detail."
    return {
        "name": name[:256],
        "value": text[:1024],
        "inline": False,
    }


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


def _money(value: Any) -> str:
    return f"${_float(value):,.2f}"


def _redact_account(account_id: Any) -> str:
    if not account_id:
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"


if __name__ == "__main__":
    raise SystemExit(main())
