#!/usr/bin/env python3
"""Approve or reject a dry-run paper order ticket without submitting it."""

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
from oqp.brokers import get_broker_profile_config  # noqa: E402
from oqp.config import load_settings  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    PaperOrderTicketStatus,
    default_paper_trading_ledger_path,
    set_paper_order_ticket_approval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Approve or reject a paper dry-run order ticket. This only updates "
            "the local ledgers and never submits an IBKR order."
        ),
    )
    parser.add_argument("order_id", help="Paper order ticket ID.")
    parser.add_argument(
        "--status",
        required=True,
        choices=[
            PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
            PaperOrderTicketStatus.REJECTED.value,
        ],
        help="Human decision for the dry-run ticket.",
    )
    parser.add_argument(
        "--approved-by",
        "--decided-by",
        dest="decided_by",
        default=os.getenv("USER") or "manual",
        help="Human or process recording the decision.",
    )
    parser.add_argument(
        "--reason",
        default="manual paper ticket decision",
        help="Short approval/rejection reason.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_paper_trading_ledger_path()),
        help="SQLite paper trading ledger path.",
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="Unified account ledger path for approval events.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Post the approval/rejection to the configured Discord webhook.",
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
    broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
    account_ledger_path = Path(args.account_ledger_path)
    account_id = broker_config.account_id or _latest_account_id(
        account_ledger_path,
        environment="paper",
        profile=str(broker_config.metadata.get("profile") or "ibkr_paper_readonly"),
    )

    approval = set_paper_order_ticket_approval(
        order_id=args.order_id,
        status=args.status,
        paper_ledger_path=Path(args.db_path),
        account_ledger_path=account_ledger_path,
        broker_config=broker_config,
        account_id=account_id,
        decided_by=args.decided_by,
        reason=args.reason,
    )
    payload = {
        "status": "updated",
        "approval": approval.to_dict(),
        "decision_by": args.decided_by,
        "reason": args.reason,
        "broker_submit_enabled": False,
    }
    if args.notify:
        _post_discord(payload, env_file=Path(args.env_file))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_result(payload)
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


def _print_result(payload: dict[str, Any]) -> None:
    approval = payload["approval"]
    print(
        f"{approval['new_status'].upper():19} {approval['order_id']} "
        f"previous={approval['previous_status']}"
    )
    print(f"        account_event={approval['account_event_id']}")
    print("        broker_submit_enabled=false")
    print(f"        {approval['message']}")


def _post_discord(payload: dict[str, Any], *, env_file: Path) -> None:
    url = _webhook_url(env_file)
    if not url:
        print("WARN   discord  No paper Discord webhook configured.", file=sys.stderr)
        return
    body = json.dumps(_discord_payload(payload), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-Paper-Ticket-Approval/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   discord  Could not post paper ticket approval: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(
            f"WARN   discord  Could not post paper ticket approval: {exc}",
            file=sys.stderr,
        )


def _discord_payload(payload: dict[str, Any]) -> dict[str, Any]:
    approval = payload["approval"]
    status = str(approval["new_status"])
    approved = status == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value
    fields = [
        _discord_field("Ticket", approval["order_id"]),
        _discord_field("Decision", status.upper()),
        _discord_field("Previous Status", approval["previous_status"]),
        _discord_field("Decision By", payload.get("decision_by") or "n/a"),
        _discord_field("Reason", payload.get("reason") or "n/a"),
        _discord_field("Account Event", approval["account_event_id"]),
        _discord_field("Broker Submit Enabled", "false"),
    ]
    return {
        "username": "OQP Paper Ticket Approval",
        "content": f"Paper ticket decision: {status.upper()}",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Paper Order Ticket Approval",
                "description": approval["message"],
                "color": 0x2ECC71 if approved else 0xE74C3C,
                "fields": fields,
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
