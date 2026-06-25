from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from oqp.accounts import load_account_trade_events
from oqp.brokers import get_broker_profile_config
from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution import OrderIntent, TradeProposal
from oqp.paper_trading import (
    PaperExecutionPolicy,
    PaperOrderTicketStatus,
    create_dry_run_order_tickets,
    load_latest_paper_orders,
    review_paper_execution_proposal,
    set_paper_order_ticket_approval,
)


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


def proposal() -> TradeProposal:
    return TradeProposal(
        proposal_id="proposal-001",
        source="unit_test",
        strategy_id="strategy-001",
        intents=(
            OrderIntent(
                instrument=Instrument("SPY", AssetClass.ETF),
                side=OrderSide.BUY,
                quantity=2,
                order_type=OrderType.LIMIT,
                limit_price=500,
                reference_price=501,
                signal_id="signal-001",
            ),
        ),
    )


class PaperOrderRouterTests(unittest.TestCase):
    def test_ready_review_creates_dry_run_tickets_and_events(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_SYMBOLS=SPY",
                "PAPER_ALLOWED_ASSET_CLASSES=etf",
                "PAPER_MAX_ORDER_NOTIONAL=2000",
                "PAPER_MAX_DAILY_NOTIONAL=5000",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        trade_proposal = proposal()
        review = review_paper_execution_proposal(
            trade_proposal,
            settings=settings,
            broker_config=broker_config,
        )

        result = create_dry_run_order_tickets(
            trade_proposal,
            review=review,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=broker_config,
            account_id="DU123456",
            review_id="review-001",
            created_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
        )
        orders = load_latest_paper_orders(paper_db)
        events = load_account_trade_events(account_db, event_type="paper_order_ticket")

        self.assertEqual(result.status.value, "dry_run")
        self.assertEqual(result.order_count, 1)
        self.assertEqual(result.account_event_count, 1)
        self.assertEqual(orders.iloc[0]["order_id"], "paper-dryrun-proposal-001-1")
        self.assertEqual(orders.iloc[0]["status"], "dry_run")
        self.assertEqual(events.iloc[0]["event_type"], "paper_order_ticket")
        self.assertEqual(events.iloc[0]["symbol"], "SPY")
        self.assertEqual(events.iloc[0]["order_id"], "paper-dryrun-proposal-001-1")

    def test_approves_dry_run_ticket_without_broker_submission(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_SYMBOLS=SPY",
                "PAPER_ALLOWED_ASSET_CLASSES=etf",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        trade_proposal = proposal()
        review = review_paper_execution_proposal(
            trade_proposal,
            settings=settings,
            broker_config=broker_config,
        )
        create_dry_run_order_tickets(
            trade_proposal,
            review=review,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=broker_config,
            account_id="DU123456",
            review_id="review-001",
            created_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
        )

        approval = set_paper_order_ticket_approval(
            order_id="paper-dryrun-proposal-001-1",
            status=PaperOrderTicketStatus.APPROVED_FOR_SUBMIT,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=broker_config,
            account_id="DU123456",
            decided_by="unit-test",
            reason="reviewed",
            decided_at=datetime(2026, 6, 25, 12, 5, tzinfo=timezone.utc),
        )
        orders = load_latest_paper_orders(paper_db)
        events = load_account_trade_events(
            account_db,
            event_type="paper_order_ticket_approved",
        )

        self.assertEqual(approval.previous_status, "dry_run")
        self.assertEqual(approval.new_status.value, "approved_for_submit")
        self.assertIn("No broker order was submitted", approval.message)
        self.assertEqual(orders.iloc[0]["status"], "approved_for_submit")
        self.assertEqual(events.iloc[0]["event_type"], "paper_order_ticket_approved")
        self.assertEqual(events.iloc[0]["order_id"], "paper-dryrun-proposal-001-1")
        self.assertIn('"broker_submit_enabled": false', events.iloc[0]["metadata_json"])

    def test_blocked_review_creates_no_tickets(self) -> None:
        tmp, settings = settings_from_lines(["ALLOW_LIVE_TRADING=false"])
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
        trade_proposal = proposal()
        review = review_paper_execution_proposal(
            trade_proposal,
            settings=settings,
            broker_config=broker_config,
            policy=PaperExecutionPolicy(
                allow_paper_trading=False,
                allowed_asset_classes=("etf",),
            ),
        )

        result = create_dry_run_order_tickets(
            trade_proposal,
            review=review,
            paper_ledger_path=Path(tmp.name) / "paper.db",
            account_ledger_path=Path(tmp.name) / "accounts.db",
            broker_config=broker_config,
        )

        self.assertEqual(result.status.value, "blocked")
        self.assertEqual(result.order_count, 0)
        self.assertEqual(result.account_event_count, 0)


if __name__ == "__main__":
    unittest.main()
