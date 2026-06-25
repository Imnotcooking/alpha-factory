from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from oqp.accounts import load_account_trade_events
from oqp.brokers import BrokerConnectionConfig, BrokerEnvironment, get_broker_profile_config
from oqp.config import load_settings
from oqp.paper_trading import (
    PaperOrderTicketStatus,
    PaperSubmissionDecision,
    record_paper_submission_preflight,
    review_paper_order_submission,
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
        )
        failed_checks = {
            check.name
            for check in preflight.checks
            if not check.passed and check.severity == "block"
        }

        self.assertEqual(preflight.decision, PaperSubmissionDecision.BLOCKED)
        self.assertEqual(failed_checks, {"Broker placement implementation"})

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


if __name__ == "__main__":
    unittest.main()
