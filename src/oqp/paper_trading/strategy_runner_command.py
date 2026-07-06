#!/usr/bin/env python3
"""Scan paper-running strategy proposals and create dry-run tickets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


from oqp.accounts import default_account_ledger_path, latest_account_id  # noqa: E402
from oqp.brokers import get_broker_profile_config  # noqa: E402
from oqp.config import REPO_ROOT, load_settings  # noqa: E402
from oqp.execution import trade_proposal_directory  # noqa: E402
from oqp.ops.notifications import (  # noqa: E402
    discord_field,
    first_env_file_value,
    post_json_webhook,
)
from oqp.paper_trading import (  # noqa: E402
    default_paper_trading_ledger_path,
    run_paper_strategy_runner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review trade proposals for strategies already marked paper_running, "
            "then create dry-run paper order tickets. No broker orders are sent."
        ),
    )
    parser.add_argument(
        "proposal_path",
        nargs="?",
        default=None,
        help="Proposal JSON file or directory. Defaults to runtime trade proposals.",
    )
    parser.add_argument(
        "--db-path",
        default=str(default_paper_trading_ledger_path()),
        help="SQLite paper trading ledger path.",
    )
    parser.add_argument(
        "--account-ledger-path",
        default=str(default_account_ledger_path()),
        help="Unified account ledger path for paper runner events.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Maximum proposal files to load when proposal_path is a directory.",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Re-review proposals that already have a paper execution review.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Post a runner summary to the configured Discord webhook.",
    )
    parser.add_argument(
        "--notify-on-action",
        action="store_true",
        help=(
            "Post only when at least one proposal is reviewed. Useful for timers "
            "that should stay quiet when no proposal artifacts are present."
        ),
    )
    parser.add_argument(
        "--require-tickets",
        action="store_true",
        help="Exit nonzero when no dry-run tickets are created.",
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
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_readonly")
    account_id = broker_config.account_id or latest_account_id(
        account_ledger_path,
        environment="paper",
        profile=profile,
    )
    proposal_path = (
        Path(args.proposal_path)
        if args.proposal_path is not None
        else trade_proposal_directory(settings)
    )

    result = run_paper_strategy_runner(
        proposal_path,
        settings=settings,
        paper_ledger_path=Path(args.db_path),
        account_ledger_path=account_ledger_path,
        broker_config=broker_config,
        account_id=account_id,
        max_files=args.max_files,
        skip_reviewed=not args.include_reviewed,
    )
    payload = {
        **result.to_dict(),
        "account_ledger_path": str(account_ledger_path),
        "paper_ledger_path": str(Path(args.db_path)),
        "broker_profile": profile,
        "broker_submit_enabled": False,
    }
    should_notify = args.notify or (
        args.notify_on_action and result.reviewed_count > 0
    )
    if should_notify:
        _post_discord(payload, env_file=Path(args.env_file))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_result(payload)

    if args.require_tickets and payload["ticket_count"] == 0:
        return 1
    return 0


def _print_result(result: dict[str, Any]) -> None:
    if result["loaded_count"] == 0:
        print(f"No trade proposals found in {result['proposal_path']}.")
        print("broker_submit_enabled=false")
        return

    print(
        "Paper strategy runner: "
        f"loaded={result['loaded_count']} "
        f"reviewed={result['reviewed_count']} "
        f"skipped={result['skipped_count']} "
        f"dry_run_tickets={result['ticket_count']}"
    )
    for item in result["items"]:
        print(
            f"{item['action'].upper():8} {item['proposal_id']} "
            f"tickets={item.get('ticket_result', {}).get('order_count', 0) if item.get('ticket_result') else 0}"
        )
        if item.get("strategy_gate"):
            gate = item["strategy_gate"]
            print(f"        strategy_gate={gate['message']}")
        if item.get("review"):
            review = item["review"]
            print(
                f"        safety={review['decision']} "
                f"notional={_money(review['estimated_notional'])}"
            )
        print(f"        {item['message']}")
    print("broker_submit_enabled=false")


def _post_discord(result: dict[str, Any], *, env_file: Path) -> None:
    url = _webhook_url(env_file)
    if not url:
        return
    post_json_webhook(
        url,
        _discord_payload(result),
        user_agent="OQP-Paper-Strategy-Runner/1.0",
        label="discord  Could not post paper strategy runner:",
    )


def _discord_payload(result: dict[str, Any]) -> dict[str, Any]:
    reviewed = [
        item for item in result["items"]
        if item.get("action") == "reviewed"
    ]
    skipped = [
        item for item in result["items"]
        if item.get("action") == "skipped"
    ]
    fields = [
        _discord_field("Loaded Proposals", result["loaded_count"]),
        _discord_field("Reviewed", result["reviewed_count"]),
        _discord_field("Skipped", result["skipped_count"]),
        _discord_field("Dry-Run Tickets", result["ticket_count"]),
        _discord_field("Broker Submit Enabled", "false"),
    ]
    for item in reviewed[:3]:
        review = item.get("review") or {}
        fields.append(
            _discord_field(
                f"REVIEWED: {item['proposal_id']}",
                (
                    f"{review.get('decision', 'unknown')} | "
                    f"tickets={item.get('ticket_result', {}).get('order_count', 0) if item.get('ticket_result') else 0} | "
                    f"{item['message']}"
                ),
            )
        )
    for item in skipped[:3]:
        fields.append(_discord_field(f"SKIPPED: {item['proposal_id']}", item["message"]))

    blocked_reviews = [
        item for item in reviewed
        if (item.get("review") or {}).get("decision") != "ready"
    ]
    color = 0xE67E22 if blocked_reviews else 0x2ECC71
    return {
        "username": "OQP Paper Strategy Runner",
        "content": "Paper strategy runner completed",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Paper Strategy Runner",
                "description": (
                    "Eligible paper-running proposals were reviewed and converted "
                    "to dry-run tickets only. No broker orders were submitted."
                ),
                "color": color,
                "fields": fields[:10],
            }
        ],
    }


def _discord_field(name: str, value: Any) -> dict[str, Any]:
    return discord_field(name, value or "n/a")


def _webhook_url(env_file: Path) -> str | None:
    for name in (
        "OQP_PAPER_DISCORD_WEBHOOK_URL",
        "OQP_DISCORD_WEBHOOK_URL",
        "OQP_HEALTH_WEBHOOK_URL",
    ):
        value = os.getenv(name)
        if value:
            return value
    return first_env_file_value(
        env_file,
        (
            "OQP_PAPER_DISCORD_WEBHOOK_URL",
            "OQP_DISCORD_WEBHOOK_URL",
            "OQP_HEALTH_WEBHOOK_URL",
        ),
    )


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
