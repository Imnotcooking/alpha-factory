from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from oqp.accounts import load_account_trade_events
from oqp.brokers import get_broker_profile_config
from oqp.config import load_settings
from oqp.contracts import CandidateStatus, StrategyCandidate
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution import OrderIntent, TradeProposal, write_trade_proposal_artifact
from oqp.paper_trading import (
    PaperStrategyStatus,
    load_latest_paper_execution_reviews,
    load_latest_paper_orders,
    run_paper_strategy_runner,
    upsert_paper_strategy_from_candidate,
)


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


def candidate(strategy_id: str = "strategy-running") -> StrategyCandidate:
    return StrategyCandidate(
        candidate_id=f"candidate-{strategy_id}",
        strategy_id=strategy_id,
        source="unit_test",
        promotion_status=CandidateStatus.PAPER_CANDIDATE,
        native_market_vertical="EQUITY_US",
        tested_market_vertical="EQUITY_US",
        target_market_vertical="EQUITY_US",
    )


def proposal(
    *,
    proposal_id: str = "proposal-running",
    strategy_id: str = "strategy-running",
    symbol: str = "SPY",
) -> TradeProposal:
    return TradeProposal(
        proposal_id=proposal_id,
        source="unit_test",
        strategy_id=strategy_id,
        intents=(
            OrderIntent(
                instrument=Instrument(symbol, AssetClass.ETF),
                side=OrderSide.BUY,
                quantity=2,
                order_type=OrderType.LIMIT,
                limit_price=500,
                reference_price=501,
                signal_id=f"signal-{proposal_id}",
            ),
        ),
    )


def register_strategy(db_path: Path, strategy_id: str = "strategy-running") -> None:
    upsert_paper_strategy_from_candidate(
        db_path,
        candidate(strategy_id),
        status=PaperStrategyStatus.RUNNING,
        max_order_notional=2_000,
        max_daily_notional=5_000,
        allowed_symbols=("SPY",),
        approved_by="unit-test",
    )


class PaperStrategyRunnerTests(unittest.TestCase):
    def test_runner_skips_unregistered_strategy_and_tickets_running_strategy(self) -> None:
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
        proposal_dir = Path(tmp.name) / "proposals"
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        register_strategy(paper_db)
        write_trade_proposal_artifact(
            proposal(proposal_id="proposal-running"),
            proposal_dir,
        )
        write_trade_proposal_artifact(
            proposal(
                proposal_id="proposal-unregistered",
                strategy_id="strategy-missing",
            ),
            proposal_dir,
        )
        broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)

        result = run_paper_strategy_runner(
            proposal_dir,
            settings=settings,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=broker_config,
            account_id="DU123456",
        )
        reviews = load_latest_paper_execution_reviews(paper_db)
        orders = load_latest_paper_orders(paper_db)
        review_events = load_account_trade_events(account_db, event_type="paper_review")
        ticket_events = load_account_trade_events(
            account_db,
            event_type="paper_order_ticket",
        )

        self.assertEqual(result.loaded_count, 2)
        self.assertEqual(result.reviewed_count, 1)
        self.assertEqual(result.skipped_count, 1)
        self.assertEqual(result.ticket_count, 1)
        self.assertEqual(set(reviews["proposal_id"]), {"proposal-running"})
        self.assertEqual(orders.iloc[0]["order_id"], "paper-dryrun-proposal-running-1")
        self.assertEqual(len(review_events), 1)
        self.assertEqual(len(ticket_events), 1)
        skipped = [item for item in result.items if item.action == "skipped"][0]
        self.assertIn("Strategy registered", skipped.message)

    def test_runner_skips_previously_reviewed_proposals_by_default(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=true",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_SYMBOLS=SPY",
                "PAPER_ALLOWED_ASSET_CLASSES=etf",
            ]
        )
        self.addCleanup(tmp.cleanup)
        proposal_dir = Path(tmp.name) / "proposals"
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        register_strategy(paper_db)
        write_trade_proposal_artifact(proposal(), proposal_dir)

        first = run_paper_strategy_runner(
            proposal_dir,
            settings=settings,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=get_broker_profile_config("ibkr_paper_readonly", settings=settings),
        )
        second = run_paper_strategy_runner(
            proposal_dir,
            settings=settings,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=get_broker_profile_config("ibkr_paper_readonly", settings=settings),
        )

        self.assertEqual(first.reviewed_count, 1)
        self.assertEqual(first.ticket_count, 1)
        self.assertEqual(second.reviewed_count, 0)
        self.assertEqual(second.skipped_count, 1)
        self.assertEqual(second.ticket_count, 0)
        self.assertIn("already has", second.items[0].message)

    def test_runner_records_review_but_no_tickets_when_safety_blocks(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "ALLOW_PAPER_TRADING=false",
                "ALLOW_LIVE_TRADING=false",
                "PAPER_ALLOWED_SYMBOLS=SPY",
                "PAPER_ALLOWED_ASSET_CLASSES=etf",
            ]
        )
        self.addCleanup(tmp.cleanup)
        proposal_dir = Path(tmp.name) / "proposals"
        paper_db = Path(tmp.name) / "paper.db"
        account_db = Path(tmp.name) / "accounts.db"
        register_strategy(paper_db)
        write_trade_proposal_artifact(proposal(), proposal_dir)

        result = run_paper_strategy_runner(
            proposal_dir,
            settings=settings,
            paper_ledger_path=paper_db,
            account_ledger_path=account_db,
            broker_config=get_broker_profile_config("ibkr_paper_readonly", settings=settings),
        )
        reviews = load_latest_paper_execution_reviews(paper_db)
        orders = load_latest_paper_orders(paper_db)
        review_events = load_account_trade_events(account_db, event_type="paper_review")

        self.assertEqual(result.reviewed_count, 1)
        self.assertEqual(result.ticket_count, 0)
        self.assertEqual(result.items[0].ticket_result.status.value, "blocked")
        self.assertEqual(reviews.iloc[0]["decision"], "blocked")
        self.assertTrue(orders.empty)
        self.assertEqual(len(review_events), 1)


if __name__ == "__main__":
    unittest.main()
