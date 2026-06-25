#!/usr/bin/env python3
"""Review paper trade proposal artifacts against the paper safety policy."""

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

from oqp.brokers import get_broker_profile_config  # noqa: E402
from oqp.accounts import (  # noqa: E402
    account_trade_events_from_proposal_review,
    default_account_ledger_path,
    load_latest_account_nav,
    write_account_trade_events,
)
from oqp.config import load_settings  # noqa: E402
from oqp.execution import (  # noqa: E402
    TradeProposal,
    load_trade_proposal_artifacts,
    parse_trade_proposal,
    trade_proposal_directory,
)
from oqp.paper_trading import (  # noqa: E402
    create_dry_run_order_tickets,
    default_paper_trading_ledger_path,
    paper_order_notional_today,
    review_paper_execution_proposal,
    write_paper_execution_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review paper trade proposal safety without placing orders.",
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
        help="Unified account ledger path for paper review events.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to runtime .env file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum proposal files to load when proposal_path is a directory.",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Post each review to the configured Discord webhook.",
    )
    parser.add_argument(
        "--create-dry-run-tickets",
        action="store_true",
        help=(
            "For READY reviews, write dry-run paper order tickets and account "
            "events. This does not submit anything to IBKR."
        ),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero if any reviewed proposal is blocked.",
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
    db_path = Path(args.db_path)
    account_ledger_path = Path(args.account_ledger_path)
    proposal_path = (
        Path(args.proposal_path)
        if args.proposal_path is not None
        else trade_proposal_directory(settings)
    )

    proposals = _load_proposals(proposal_path, max_files=args.max_files)
    broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
    daily_used = paper_order_notional_today(db_path)
    account_id = broker_config.account_id or _latest_account_id(
        account_ledger_path,
        environment="paper",
        profile=str(broker_config.metadata.get("profile") or "ibkr_paper_readonly"),
    )

    reviews = []
    for proposal in proposals:
        review = review_paper_execution_proposal(
            proposal,
            settings=settings,
            broker_config=broker_config,
            daily_notional_used=daily_used,
        )
        write_result = write_paper_execution_review(
            db_path,
            proposal_id=proposal.proposal_id,
            decision=review.decision.value,
            checks=[check.to_dict() for check in review.checks],
            estimated_notional=review.estimated_notional,
            order_count=review.order_count,
            message=review.message,
            reviewed_at=review.reviewed_at,
        )
        event_result = write_account_trade_events(
            account_ledger_path,
            account_trade_events_from_proposal_review(
                proposal,
                decision=review.decision.value,
                reviewed_at=review.reviewed_at,
                environment="paper",
                profile=str(
                    broker_config.metadata.get("profile") or "ibkr_paper_readonly"
                ),
                broker=broker_config.broker,
                account_id=account_id,
                review_id=write_result.review_id,
                message=review.message,
            ),
        )
        ticket_result = None
        if args.create_dry_run_tickets:
            ticket_result = create_dry_run_order_tickets(
                proposal,
                review=review,
                paper_ledger_path=db_path,
                account_ledger_path=account_ledger_path,
                broker_config=broker_config,
                account_id=account_id,
                review_id=write_result.review_id,
                created_at=review.reviewed_at,
            )
        payload = {
            "proposal": proposal.proposal_id,
            "review": review.to_dict(),
            "write": write_result.to_dict(),
            "account_events": event_result.to_dict(),
            "order_tickets": None if ticket_result is None else ticket_result.to_dict(),
        }
        reviews.append(payload)
        if args.notify:
            _post_discord(payload)

    result = {
        "status": "reviewed",
        "proposal_path": str(proposal_path),
        "reviews": reviews,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_reviews(result)

    if args.require_ready and any(
        review["review"]["decision"] != "ready" for review in reviews
    ):
        return 1
    return 0


def _load_proposals(path: Path, *, max_files: int) -> list[TradeProposal]:
    if path.is_file():
        return [parse_trade_proposal(json.loads(path.read_text(encoding="utf-8")))]
    result = load_trade_proposal_artifacts(path, max_files=max_files)
    if result.issues:
        issue_text = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        raise SystemExit(f"Proposal artifact issues: {issue_text}")
    return [loaded.proposal for loaded in result.loaded]


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


def _print_reviews(result: dict[str, Any]) -> None:
    reviews = result["reviews"]
    if not reviews:
        print(f"No trade proposals found in {result['proposal_path']}.")
        return
    for item in reviews:
        review = item["review"]
        print(
            f"{review['decision'].upper():7} {review['proposal_id']} "
            f"orders={review['order_count']} "
            f"notional={_money(review['estimated_notional'])}"
        )
        account_events = item.get("account_events", {})
        if account_events:
            print(f"        account_events={account_events.get('event_count', 0)}")
        order_tickets = item.get("order_tickets")
        if order_tickets:
            print(
                f"        dry_run_tickets={order_tickets.get('order_count', 0)} "
                f"status={order_tickets.get('status')}"
            )
        print(f"        {review['message']}")
        for check in review["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(
                f"        {status:4} {check['severity']:<7} "
                f"{check['name']}: {check['detail']}"
            )


def _post_discord(payload: dict[str, Any]) -> None:
    url = (
        os.getenv("OQP_PAPER_DISCORD_WEBHOOK_URL")
        or os.getenv("OQP_DISCORD_WEBHOOK_URL")
        or os.getenv("OQP_HEALTH_WEBHOOK_URL")
    )
    if not url:
        print("WARN   discord  No paper Discord webhook configured.", file=sys.stderr)
        return
    body = json.dumps(_discord_payload(payload), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-Paper-Execution-Review/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   discord  Could not post paper review: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARN   discord  Could not post paper review: {exc}", file=sys.stderr)


def _discord_payload(payload: dict[str, Any]) -> dict[str, Any]:
    review = payload["review"]
    account_events = payload.get("account_events", {})
    order_tickets = payload.get("order_tickets") or {}
    failed = [
        check for check in review["checks"]
        if not check["passed"] and check["severity"] == "block"
    ]
    fields = [
        _discord_field("Proposal", review["proposal_id"]),
        _discord_field("Decision", review["decision"].upper()),
        _discord_field("Orders", str(review["order_count"])),
        _discord_field("Estimated Notional", _money(review["estimated_notional"])),
        _discord_field("Account Events", str(account_events.get("event_count", 0))),
        _discord_field("Dry-Run Tickets", str(order_tickets.get("order_count", 0))),
        _discord_field("Message", review["message"]),
    ]
    for check in failed[:5]:
        fields.append(_discord_field(f"BLOCK: {check['name']}", check["detail"]))

    return {
        "username": "OQP Paper Execution Review",
        "content": f"Paper execution review: {review['decision'].upper()}",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "Paper Execution Safety Review",
                "description": review["message"],
                "color": 0x2ECC71 if review["decision"] == "ready" else 0xE67E22,
                "timestamp": review["reviewed_at"],
                "fields": fields[:10],
            }
        ],
    }


def _discord_field(name: str, value: str) -> dict[str, Any]:
    text = str(value).strip() or "n/a"
    return {"name": name[:256], "value": text[:1024], "inline": False}


def _money(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
