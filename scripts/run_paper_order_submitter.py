#!/usr/bin/env python3
"""Run paper order submission preflight checks or guarded paper submissions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import default_account_ledger_path, load_latest_account_nav  # noqa: E402
from oqp.brokers import get_broker_adapter, get_broker_profile_config  # noqa: E402
from oqp.config import load_settings  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    PaperOrderTicketStatus,
    default_paper_trading_ledger_path,
    load_latest_paper_orders,
    load_paper_strategy_record,
    record_paper_submission_preflight,
    review_paper_order_submission,
    submit_approved_paper_order_ticket,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review approved paper order tickets for submission readiness. "
            "Use --submit-approved to place guarded IBKR paper orders."
        ),
    )
    parser.add_argument(
        "--db-path",
        default=str(default_paper_trading_ledger_path()),
        help="SQLite paper trading ledger path.",
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="Unified account ledger path for preflight events.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum recent paper tickets to scan.",
    )
    parser.add_argument(
        "--record-events",
        action="store_true",
        help="Write preflight results to the unified account event ledger.",
    )
    parser.add_argument(
        "--submit-approved",
        action="store_true",
        help=(
            "Submit approved, preflight-ready tickets to the IBKR paper profile. "
            "Requires ALLOW_PAPER_ORDER_SUBMIT=true and the paper submit profile."
        ),
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Post a preflight summary to the configured Discord webhook.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero if any reviewed ticket is blocked.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env_file)
    broker_config = get_broker_profile_config(
        "ibkr_paper_submit" if args.submit_approved else "ibkr_paper_readonly",
        settings=settings,
    )
    account_ledger_path = Path(args.account_ledger_path)
    account_id = broker_config.account_id or _latest_account_id(
        account_ledger_path,
        environment="paper",
        profile=str(broker_config.metadata.get("profile") or "ibkr_paper_readonly"),
    )
    orders = load_latest_paper_orders(Path(args.db_path), limit=args.limit)
    approved = (
        orders[orders["status"].eq(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value)]
        if not orders.empty and "status" in orders.columns
        else orders.iloc[0:0]
    )

    preflights: list[dict[str, Any]] = []
    submissions: list[dict[str, Any]] = []
    for row in approved.to_dict("records"):
        strategy_record = load_paper_strategy_record(
            Path(args.db_path),
            str(row.get("strategy_id") or ""),
        )
        if args.submit_approved:
            submission = submit_approved_paper_order_ticket(
                order_id=str(row.get("order_id") or ""),
                paper_ledger_path=Path(args.db_path),
                account_ledger_path=account_ledger_path,
                settings=settings,
                broker_config=broker_config,
                broker=get_broker_adapter("ibkr", settings=settings),
                strategy_record=strategy_record,
                account_id=account_id,
            )
            submissions.append(submission.to_dict())
            preflights.append(
                {
                    "preflight": submission.preflight.to_dict(),
                    "record": None,
                    "submission": submission.to_dict(),
                }
            )
        else:
            preflight = review_paper_order_submission(
                row,
                settings=settings,
                broker_config=broker_config,
                strategy_record=strategy_record,
            )
            record = None
            if args.record_events:
                record = record_paper_submission_preflight(
                    account_ledger_path,
                    preflight,
                    broker_config=broker_config,
                    account_id=account_id,
                )
            preflights.append(
                {
                    "preflight": preflight.to_dict(),
                    "record": None if record is None else record.to_dict(),
                }
            )

    result = {
        "status": "reviewed",
        "paper_ledger_path": str(Path(args.db_path)),
        "account_ledger_path": str(account_ledger_path),
        "approved_ticket_count": len(approved),
        "record_events": bool(args.record_events),
        "broker_submit_enabled": bool(args.submit_approved),
        "preflights": preflights,
        "submissions": submissions,
    }
    if args.notify:
        _post_discord(result, env_file=Path(args.env_file))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_result(result)

    if args.require_ready and any(
        item["preflight"]["decision"] != "ready" for item in preflights
    ):
        return 1
    return 0


def _latest_account_id(
    account_ledger_path: Path,
    *,
    environment: str,
    profile: str,
) -> str | None:
    nav = load_latest_account_nav(
        account_ledger_path,
        environment=environment,
        profile=profile,
    )
    if nav.empty:
        return None
    value = nav.iloc[0].get("account_id")
    text = str(value).strip() if value is not None else ""
    return text or None


def _print_result(result: dict[str, Any]) -> None:
    if not result["preflights"]:
        print("No approved paper tickets found for submission preflight.")
        print("broker_submit_enabled=false")
        return

    for item in result["preflights"]:
        preflight = item["preflight"]
        print(
            f"{preflight['decision'].upper():7} {preflight['order_id']} "
            f"{preflight['symbol']} {preflight.get('side') or ''}"
        )
        record = item.get("record")
        if record:
            print(f"        account_event={record['event_id']}")
        print(f"        {preflight['message']}")
        for check in preflight["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(
                f"        {status:4} {check['severity']:<7} "
                f"{check['name']}: {check['detail']}"
            )
    print(f"broker_submit_enabled={str(result.get('broker_submit_enabled', False)).lower()}")


def _post_discord(result: dict[str, Any], *, env_file: Path) -> None:
    url = _webhook_url(env_file)
    if not url:
        print("WARN   discord  No paper Discord webhook configured.", file=sys.stderr)
        return
    body = json.dumps(_discord_payload(result), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-Paper-Order-Submitter/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   discord  Could not post paper submitter preflight: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(
            f"WARN   discord  Could not post paper submitter preflight: {exc}",
            file=sys.stderr,
        )


def _discord_payload(result: dict[str, Any]) -> dict[str, Any]:
    preflights = [item["preflight"] for item in result["preflights"]]
    submissions = result.get("submissions", [])
    blocked = [item for item in preflights if item["decision"] != "ready"]
    fields = [
        _discord_field("Approved Tickets", result["approved_ticket_count"]),
        _discord_field("Blocked", len(blocked)),
        _discord_field("Recorded Events", result["record_events"]),
        _discord_field("Broker Submit Enabled", result["broker_submit_enabled"]),
        _discord_field("Submitted", len(submissions)),
    ]
    for preflight in preflights[:4]:
        fields.append(
            _discord_field(
                f"{preflight['decision'].upper()}: {preflight['order_id']}",
                preflight["message"],
            )
        )
    return {
        "username": "OQP Paper Submitter",
        "content": "Paper submitter preflight completed",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Paper Order Submission Preflight",
                "description": (
                    "IBKR paper orders submitted only when --submit-approved is used."
                ),
                "color": 0xE67E22 if blocked else 0x2ECC71,
                "fields": fields[:10],
            }
        ],
    }


def _discord_field(name: str, value: Any) -> dict[str, Any]:
    text = str(value).strip() or "n/a"
    return {"name": name[:256], "value": text[:1024], "inline": False}


def _webhook_url(env_file: Path) -> str | None:
    for name in (
        "OQP_PAPER_DISCORD_WEBHOOK_URL",
        "OQP_DISCORD_WEBHOOK_URL",
        "OQP_HEALTH_WEBHOOK_URL",
    ):
        value = os.getenv(name) or _env_file_value(env_file, name)
        if value:
            return value
    return None


def _env_file_value(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
