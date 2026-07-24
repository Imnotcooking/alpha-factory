from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from oqp.contracts import CandidateStatus, StrategyCandidate
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution import OrderIntent, TradeProposal
from oqp.paper_trading import (
    PaperStrategyStatus,
    is_paper_strategy_running,
    load_paper_strategy_record,
    load_paper_strategy_registry,
    review_paper_strategy_gate,
    upsert_paper_strategy_from_candidate,
)


def paper_candidate(**overrides) -> StrategyCandidate:
    values = {
        "candidate_id": "candidate-run-001",
        "strategy_id": "fac_demo",
        "source": "unit_test",
        "promotion_status": CandidateStatus.PAPER_CANDIDATE,
        "native_market_vertical": "EQUITY_US",
        "tested_market_vertical": "EQUITY_US",
        "target_market_vertical": "EQUITY_US",
        "research_run_id": "run-001",
    }
    values.update(overrides)
    return StrategyCandidate(**values)


def trade_proposal(**overrides) -> TradeProposal:
    values = {
        "proposal_id": "proposal-001",
        "source": "unit_test",
        "strategy_id": "fac_demo",
        "intents": (
            OrderIntent(
                instrument=Instrument("SPY", AssetClass.ETF),
                side=OrderSide.BUY,
                quantity=1,
                order_type=OrderType.LIMIT,
                limit_price=500,
                reference_price=500,
            ),
        ),
    }
    values.update(overrides)
    return TradeProposal(**values)


class PaperStrategyRegistryTests(unittest.TestCase):
    def test_registers_paper_running_strategy_from_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            candidate = paper_candidate()

            result = upsert_paper_strategy_from_candidate(
                db_path,
                candidate,
                status=PaperStrategyStatus.RUNNING,
                max_order_notional=1_000,
                max_daily_notional=5_000,
                allowed_symbols=("SPY", "AAPL"),
                rebalance_frequency="daily",
                approved_by="unit-test",
                approved_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
                source_artifact="runtime/artifacts/strategy_candidates/demo.json",
            )
            registry = load_paper_strategy_registry(db_path)
            record = load_paper_strategy_record(db_path, "fac_demo")

        self.assertEqual(result.status, PaperStrategyStatus.RUNNING)
        self.assertEqual(len(registry), 1)
        self.assertIsNotNone(record)
        self.assertEqual(record["strategy_id"], "fac_demo")
        self.assertEqual(record["status"], "paper_running")
        self.assertEqual(record["allowed_symbols"], ["AAPL", "SPY"])
        self.assertFalse(record["kill_switch"])
        self.assertTrue(is_paper_strategy_running(record))

    def test_rejects_cross_market_candidate_for_paper_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            candidate = paper_candidate(
                candidate_id="candidate-cross",
                strategy_id="fac_cross",
                tested_market_vertical="FUTURES_CN",
                target_market_vertical="EQUITY_US",
            )

            with self.assertRaises(ValueError):
                upsert_paper_strategy_from_candidate(
                    db_path,
                    candidate,
                    status=PaperStrategyStatus.RUNNING,
                )

    def test_kill_switch_disables_running_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            candidate = paper_candidate()

            upsert_paper_strategy_from_candidate(
                db_path,
                candidate,
                status=PaperStrategyStatus.RUNNING,
                kill_switch=True,
            )
            record = load_paper_strategy_record(db_path, "fac_demo")

        self.assertIsNotNone(record)
        self.assertTrue(record["kill_switch"])
        self.assertFalse(is_paper_strategy_running(record))

    def test_strategy_gate_passes_for_running_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            upsert_paper_strategy_from_candidate(
                db_path,
                paper_candidate(),
                status=PaperStrategyStatus.RUNNING,
                max_order_notional=1_000,
                allowed_symbols=("SPY",),
            )

            gate = review_paper_strategy_gate(db_path, trade_proposal())

        self.assertTrue(gate.passed)
        self.assertEqual(gate.strategy_id, "fac_demo")

    def test_strategy_gate_blocks_unregistered_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = review_paper_strategy_gate(Path(tmp) / "paper.db", trade_proposal())

        self.assertFalse(gate.passed)
        self.assertIn("Strategy registered", [check.name for check in gate.checks if not check.passed])

    def test_strategy_gate_blocks_symbol_outside_strategy_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            upsert_paper_strategy_from_candidate(
                db_path,
                paper_candidate(),
                status=PaperStrategyStatus.RUNNING,
                allowed_symbols=("AAPL",),
            )

            gate = review_paper_strategy_gate(db_path, trade_proposal())

        self.assertFalse(gate.passed)
        self.assertIn(
            "Strategy symbol allowlist",
            [check.name for check in gate.checks if not check.passed],
        )


if __name__ == "__main__":
    unittest.main()
