from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from oqp.brokers import (
    BrokerConnectionStatus,
    BrokerHealth,
    IBKRReadOnlyPortfolioSnapshot,
)
from oqp.paper_trading import (
    ensure_paper_trading_schema,
    load_latest_paper_execution_reviews,
    load_latest_paper_nav,
    load_latest_paper_positions,
    write_paper_execution_review,
    write_paper_snapshot,
)


def sample_snapshot(*, nav: float = 1_000_000.0) -> IBKRReadOnlyPortfolioSnapshot:
    return IBKRReadOnlyPortfolioSnapshot(
        health=BrokerHealth(
            broker="ibkr",
            status=BrokerConnectionStatus.CONNECTED,
            account_id="DU123456",
        ),
        metrics={
            "Total_NAV_USD": nav,
            "Available_Cash_USD": 750_000.0,
            "Margin_Buffer_USD": 800_000.0,
        },
        position_rows=(
            {
                "Ticker": "AAPL",
                "Shares": 10,
                "AvgPrice": 100,
                "Broker_Price": 120,
                "Broker_PnL": 200,
                "Currency": "USD",
                "AssetType": "Equity",
                "Multiplier": 1,
            },
            {
                "Ticker": "MSFT",
                "Shares": 5,
                "AvgPrice": 200,
                "Broker_Price": 210,
                "Broker_PnL": 50,
                "Currency": "USD",
                "AssetType": "Equity",
                "Multiplier": 1,
            },
        ),
    )


class PaperTradingLedgerTests(unittest.TestCase):
    def test_schema_creates_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            ensure_paper_trading_schema(db_path)

            import sqlite3

            with sqlite3.connect(db_path) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

        self.assertIn("paper_account_snapshots", tables)
        self.assertIn("paper_positions", tables)
        self.assertIn("paper_nav", tables)
        self.assertIn("paper_orders", tables)
        self.assertIn("paper_fills", tables)
        self.assertIn("paper_execution_reviews", tables)

    def test_writes_snapshot_positions_and_daily_nav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            first = write_paper_snapshot(
                db_path,
                sample_snapshot(nav=1_000_000),
                snapshot_date="2026-06-24",
            )
            second = write_paper_snapshot(
                db_path,
                sample_snapshot(nav=1_010_000),
                snapshot_date="2026-06-25",
            )

            latest_nav = load_latest_paper_nav(db_path)
            latest_positions = load_latest_paper_positions(db_path)

        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertEqual(first.position_rows, 2)
        self.assertEqual(second.daily_pnl, 10_000)
        self.assertEqual(latest_nav.iloc[0]["date"], "2026-06-25")
        self.assertEqual(float(latest_nav.iloc[0]["net_liquidation"]), 1_010_000)
        self.assertEqual(len(latest_positions), 2)

    def test_rejects_error_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            snapshot = IBKRReadOnlyPortfolioSnapshot(
                health=BrokerHealth(
                    broker="ibkr",
                    status=BrokerConnectionStatus.ERROR,
                    account_id="DU123456",
                ),
                error="could not connect",
            )

            with self.assertRaises(ValueError):
                write_paper_snapshot(db_path, snapshot)

    def test_writes_execution_review_audit_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            result = write_paper_execution_review(
                db_path,
                proposal_id="proposal-001",
                decision="blocked",
                checks=[
                    {
                        "name": "Paper trading switch",
                        "passed": False,
                        "severity": "block",
                        "detail": "ALLOW_PAPER_TRADING=false",
                    }
                ],
                estimated_notional=1_000,
                order_count=1,
                message="Blocked by switch",
            )

            import sqlite3

            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    "SELECT proposal_id, decision, order_count FROM paper_execution_reviews"
                ).fetchone()

        self.assertEqual(result.proposal_id, "proposal-001")
        self.assertEqual(row, ("proposal-001", "blocked", 1))

    def test_loads_latest_execution_review_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            write_paper_execution_review(
                db_path,
                proposal_id="proposal-old",
                decision="blocked",
                checks=[],
                estimated_notional=100,
                order_count=1,
                message="old",
                reviewed_at=datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc),
            )
            write_paper_execution_review(
                db_path,
                proposal_id="proposal-new",
                decision="ready",
                checks=[],
                estimated_notional=200,
                order_count=2,
                message="new",
                reviewed_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
            )

            reviews = load_latest_paper_execution_reviews(db_path, limit=1)

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews.iloc[0]["proposal_id"], "proposal-new")
        self.assertEqual(reviews.iloc[0]["decision"], "ready")


if __name__ == "__main__":
    unittest.main()
