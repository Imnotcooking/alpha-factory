from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from oqp.accounts import load_account_trade_events
from oqp.brokers import (
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    BrokerEnvironment,
    BrokerHealth,
    OrderReceipt,
    get_broker_profile_config,
)
from oqp.config import load_settings
from oqp.domain import OrderStatus
from oqp.paper_trading import (
    PaperOrderTicketStatus,
    PaperSubmissionDecision,
    PaperStrategyStatus,
    load_paper_order_ticket,
    record_paper_submission_preflight,
    review_paper_order_submission,
    submit_approved_paper_order_ticket,
    write_paper_order_tickets,
)


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


def approved_ticket() -> dict[str, object]:
    return {
        "order_id": "paper-dryrun-proposal-001-1",
        "created_at": "2026-06-25T12:00:00+00:00",
        "strategy_id": "strategy-001",
        "symbol": "SPY",
        "side": "buy",
        "quantity": 2,
        "order_type": "limit",
        "limit_price": 500,
        "status": PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
        "metadata": {
            "proposal_id": "proposal-001",
            "approval_status": PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
        },
    }


class PaperOrderSubmitterTests(unittest.TestCase):
    def test_submitter_preflight_blocks_approved_ticket_by_default(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        preflight = review_paper_order_submission(
            approved_ticket(),
            settings=settings,
            broker_config=broker_config,
            checked_at=datetime(2026, 6, 25, 12, 10, tzinfo=timezone.utc),
        )
        failed_checks = {
            check.name
            for check in preflight.checks
            if not check.passed and check.severity == "block"
        }

        self.assertEqual(preflight.decision, PaperSubmissionDecision.BLOCKED)
        self.assertIn("Strategy paper-running", failed_checks)
        self.assertIn("Paper submit switch", failed_checks)
        self.assertIn("Broker profile write-enabled", failed_checks)
        self.assertIn("Broker placement implementation", failed_checks)
        self.assertFalse(preflight.metadata["broker_submit_enabled"])

    def test_submitter_preflight_still_blocks_when_switch_is_on(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_LIVE_TRADING=false",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = BrokerConnectionConfig(
            broker="ibkr",
            host="127.0.0.1",
            port=7497,
            client_id=101,
            environment=BrokerEnvironment.PAPER,
            readonly=False,
            metadata={"profile": "ibkr_paper_submit_future"},
        )

        preflight = review_paper_order_submission(
            approved_ticket(),
            settings=settings,
            broker_config=broker_config,
            strategy_record={
                "strategy_id": "strategy-001",
                "status": PaperStrategyStatus.RUNNING.value,
                "kill_switch": False,
            },
        )
        failed_checks = {
            check.name
            for check in preflight.checks
            if not check.passed and check.severity == "block"
        }

        self.assertEqual(preflight.decision, PaperSubmissionDecision.BLOCKED)
        self.assertEqual(failed_checks, {"Broker placement implementation"})

    def test_paper_submit_profile_requires_submit_switch(self) -> None:
        tmp, locked_settings = settings_from_lines(["ALLOW_LIVE_TRADING=false"])
        self.addCleanup(tmp.cleanup)

        with self.assertRaises(ValueError):
            get_broker_profile_config("ibkr_paper_submit", settings=locked_settings)

        tmp_enabled, enabled_settings = settings_from_lines(
            [
                "ALLOW_LIVE_TRADING=false",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "IBKR_PAPER_SUBMIT_CLIENT_ID=321",
            ]
        )
        self.addCleanup(tmp_enabled.cleanup)
        config = get_broker_profile_config("ibkr_paper_submit", settings=enabled_settings)

        self.assertEqual(config.environment, BrokerEnvironment.PAPER)
        self.assertFalse(config.readonly)
        self.assertEqual(config.client_id, 321)
        self.assertEqual(config.metadata["profile"], "ibkr_paper_submit")

    def test_submitter_preflight_ready_when_submitter_is_implemented(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_LIVE_TRADING=false",
            ]
        )
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_submit", settings=settings)

        preflight = review_paper_order_submission(
            approved_ticket(),
            settings=settings,
            broker_config=broker_config,
            strategy_record={
                "strategy_id": "strategy-001",
                "status": PaperStrategyStatus.RUNNING.value,
                "kill_switch": False,
            },
            broker_submit_implemented=True,
        )

        self.assertEqual(preflight.decision, PaperSubmissionDecision.READY)
        self.assertTrue(preflight.metadata["broker_submit_enabled"])

    def test_records_submitter_preflight_event_without_order_submission(self) -> None:
        tmp, settings = settings_from_lines(["ALLOW_LIVE_TRADING=false"])
        self.addCleanup(tmp.cleanup)
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
        account_db = Path(tmp.name) / "accounts.db"
        preflight = review_paper_order_submission(
            approved_ticket(),
            settings=settings,
            broker_config=broker_config,
            checked_at=datetime(2026, 6, 25, 12, 10, tzinfo=timezone.utc),
        )

        result = record_paper_submission_preflight(
            account_db,
            preflight,
            broker_config=broker_config,
            account_id="DU123456",
        )
        events = load_account_trade_events(
            account_db,
            event_type="paper_order_submission_preflight_blocked",
        )

        self.assertEqual(result.decision, PaperSubmissionDecision.BLOCKED)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["order_id"], "paper-dryrun-proposal-001-1")
        self.assertIn('"broker_submit_enabled": false', events.iloc[0]["metadata_json"])
        self.assertIn('"submitter_skeleton": true', events.iloc[0]["metadata_json"])

    def test_submits_approved_ticket_with_guarded_paper_broker(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_LIVE_TRADING=false",
            ]
        )
        self.addCleanup(tmp.cleanup)
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        broker_config = get_broker_profile_config("ibkr_paper_submit", settings=settings)
        write_paper_order_tickets(
            paper_db,
            [
                {
                    **approved_ticket(),
                    "metadata": {
                        "proposal_id": "proposal-001",
                        "approval_status": PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
                        "asset_class": "etf",
                        "currency": "USD",
                    },
                }
            ],
        )

        result = submit_approved_paper_order_ticket(
            order_id="paper-dryrun-proposal-001-1",
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            settings=settings,
            broker_config=broker_config,
            broker=FakePaperBroker(),
            strategy_record={
                "strategy_id": "strategy-001",
                "status": PaperStrategyStatus.RUNNING.value,
                "kill_switch": False,
            },
            account_id="DU123456",
            submitted_at=datetime(2026, 6, 25, 12, 20, tzinfo=timezone.utc),
        )
        ticket = load_paper_order_ticket(paper_db, "paper-dryrun-proposal-001-1")
        events = load_account_trade_events(account_db, event_type="paper_order_submitted")

        self.assertTrue(result.submitted, result.to_dict())
        self.assertEqual(result.decision, PaperSubmissionDecision.SUBMITTED)
        self.assertEqual(result.receipt.broker_order_id, "9001")
        self.assertEqual(ticket["status"], PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value)
        self.assertEqual(ticket["metadata"]["broker_order_id"], "9001")
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["broker_order_id"], "9001")
        self.assertIn('"broker_submit_enabled": true', events.iloc[0]["metadata_json"])


class FakePaperBroker:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.placed_orders = []

    def connect(self, config: BrokerConnectionConfig) -> BrokerHealth:
        self.connected = True
        return BrokerHealth(
            broker="ibkr",
            status=BrokerConnectionStatus.CONNECTED,
            account_id="DU123456",
            message="connected",
        )

    def disconnect(self) -> None:
        self.disconnected = True

    def place_order(self, order):
        self.placed_orders.append(order)
        return OrderReceipt(
            order=order,
            status=OrderStatus.SUBMITTED,
            broker_order_id="9001",
            client_order_id=order.client_order_id,
            message="Submitted",
            metadata={"fake": True},
        )


if __name__ == "__main__":
    unittest.main()
