from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from oqp.accounts import (
    AccountEnvironment,
    TradeEvent,
    account_snapshot_from_ibkr_readonly,
    account_snapshot_from_live_positions_frame,
    account_asset_summary,
    account_nav_drawdowns,
    account_positions_display,
    account_trade_event_summary,
    account_trade_events_display,
    account_trade_events_from_proposal_review,
    ensure_account_ledger_schema,
    load_account_nav_history,
    load_account_trade_events,
    load_latest_account_nav,
    load_latest_account_positions,
    write_account_trade_event,
    write_account_snapshot,
)
from oqp.brokers import (
    BrokerConnectionStatus,
    BrokerHealth,
    IBKRReadOnlyPortfolioSnapshot,
)
from oqp.domain import AssetClass, Instrument, OrderSide, OrderType
from oqp.execution import OrderIntent, TradeProposal


def sample_ibkr_snapshot(
    *,
    nav: float = 1_000_000.0,
    checked_at: datetime | None = None,
) -> IBKRReadOnlyPortfolioSnapshot:
    return IBKRReadOnlyPortfolioSnapshot(
        health=BrokerHealth(
            broker="ibkr",
            status=BrokerConnectionStatus.CONNECTED,
            account_id="DU123456",
            checked_at=checked_at or datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
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
                "Broker": "IBKR Paper",
            },
        ),
    )


class AccountLedgerTests(unittest.TestCase):
    def test_schema_creates_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.db"
            ensure_account_ledger_schema(db_path)

            with closing(sqlite3.connect(db_path)) as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }

        self.assertIn("account_snapshots", tables)
        self.assertIn("account_positions", tables)
        self.assertIn("account_cash", tables)
        self.assertIn("account_nav", tables)
        self.assertIn("account_trade_events", tables)

    def test_writes_ibkr_paper_snapshot_and_daily_nav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.db"
            first = account_snapshot_from_ibkr_readonly(
                sample_ibkr_snapshot(nav=1_000_000),
                environment="paper",
                profile="ibkr_paper_readonly",
                snapshot_date="2026-06-24",
            )
            second = account_snapshot_from_ibkr_readonly(
                sample_ibkr_snapshot(
                    nav=1_010_000,
                    checked_at=datetime(2026, 6, 25, 13, 0, tzinfo=timezone.utc),
                ),
                environment="paper",
                profile="ibkr_paper_readonly",
                snapshot_date="2026-06-25",
            )
            first_result = write_account_snapshot(
                db_path,
                first,
                snapshot_date="2026-06-24",
            )
            second_result = write_account_snapshot(
                db_path,
                second,
                snapshot_date="2026-06-25",
            )
            latest_nav = load_latest_account_nav(db_path, environment="paper")
            nav_history = load_account_nav_history(db_path, environment="paper")
            limited_history = load_account_nav_history(
                db_path,
                environment="paper",
                limit=1,
            )
            latest_positions = load_latest_account_positions(db_path, environment="paper")

        self.assertEqual(first_result.daily_pnl, 0.0)
        self.assertEqual(second_result.daily_pnl, 10_000.0)
        self.assertEqual(second_result.position_rows, 1)
        self.assertEqual(latest_nav.iloc[0]["account_id"], "DU123456")
        self.assertEqual(float(latest_nav.iloc[0]["net_liquidation"]), 1_010_000.0)
        self.assertEqual(list(nav_history["date"]), ["2026-06-24", "2026-06-25"])
        self.assertEqual(list(limited_history["date"]), ["2026-06-25"])
        self.assertEqual(latest_positions.iloc[0]["symbol"], "AAPL")
        self.assertEqual(float(latest_positions.iloc[0]["market_value"]), 1200.0)

    def test_converts_live_positions_frame_to_account_snapshot(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "Broker": "IBKR Live",
                    "Ticker": "MSFT",
                    "AssetType": "Equity",
                    "Shares": 2,
                    "AvgPrice": 300,
                    "Broker_Price": 310,
                    "Broker_PnL": 20,
                    "Currency": "USD",
                }
            ]
        )

        snapshot = account_snapshot_from_live_positions_frame(
            frame,
            metrics={"Total_NAV_USD": 40_000, "Available_Cash_USD": 1_000},
            environment="live",
            profile="ibkr_live_readonly",
            account_id="U123",
            snapshot_date="2026-06-25",
        )

        self.assertEqual(snapshot.account_key, "live:ibkr:ibkr_live_readonly:U123")
        self.assertEqual(snapshot.position_count, 1)
        self.assertEqual(snapshot.positions[0].symbol, "MSFT")
        self.assertEqual(snapshot.cash_balances[0].cash, 1_000)

    def test_account_reporting_transforms_are_dashboard_ready(self) -> None:
        nav_history = pd.DataFrame(
            [
                {
                    "date": "2026-06-24",
                    "net_liquidation": 100.0,
                    "cash": 10.0,
                    "daily_pnl": 0.0,
                    "position_count": 1,
                },
                {
                    "date": "2026-06-25",
                    "net_liquidation": 90.0,
                    "cash": 12.0,
                    "daily_pnl": -10.0,
                    "position_count": 1,
                },
            ]
        )
        positions = pd.DataFrame(
            [
                {
                    "symbol": "SPY",
                    "asset_class": "etf",
                    "quantity": 2,
                    "market_price": 500.0,
                    "market_value": 1000.0,
                    "unrealized_pnl": 25.0,
                    "currency": "USD",
                    "as_of": "2026-06-25T12:00:00+00:00",
                }
            ]
        )

        drawdowns = account_nav_drawdowns(nav_history)
        display = account_positions_display(positions)
        summary = account_asset_summary(positions)

        self.assertEqual(float(drawdowns.iloc[-1]["drawdown"]), -10.0)
        self.assertEqual(float(drawdowns.iloc[-1]["drawdown_pct"]), -0.1)
        self.assertIn("Market Value", display.columns)
        self.assertEqual(display.iloc[0]["Symbol"], "SPY")
        self.assertEqual(summary.iloc[0]["Asset Class"], "etf")
        self.assertEqual(float(summary.iloc[0]["Market Value"]), 1000.0)

    def test_writes_and_loads_account_trade_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.db"
            event = TradeEvent(
                event_id="evt-001",
                event_type="fill",
                occurred_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
                account_id="DU123456",
                broker="ibkr",
                profile="ibkr_paper_readonly",
                environment=AccountEnvironment.PAPER,
                symbol="SPY",
                side="buy",
                quantity=2,
                price=500,
                commission=1,
                currency="USD",
                strategy_id="strategy-001",
                order_id="order-001",
                broker_order_id="broker-001",
                metadata={"proposal_id": "proposal-001"},
            )

            result = write_account_trade_event(db_path, event)
            loaded = load_account_trade_events(db_path, environment="paper")
            display = account_trade_events_display(loaded)
            summary = account_trade_event_summary(loaded)

        self.assertEqual(result.event_count, 1)
        self.assertEqual(loaded.iloc[0]["event_id"], "evt-001")
        self.assertEqual(
            loaded.iloc[0]["account_key"],
            "paper:ibkr:ibkr_paper_readonly:DU123456",
        )
        self.assertEqual(
            json.loads(loaded.iloc[0]["metadata_json"])["proposal_id"],
            "proposal-001",
        )
        self.assertEqual(display.iloc[0]["Symbol"], "SPY")
        self.assertEqual(summary.iloc[0]["Event"], "fill")

    def test_converts_paper_review_to_account_trade_events(self) -> None:
        proposal = TradeProposal(
            proposal_id="proposal-001",
            source="unit_test",
            strategy_id="strategy-proposal",
            research_run_id="run-001",
            intents=(
                OrderIntent(
                    instrument=Instrument("SPY", AssetClass.ETF),
                    side=OrderSide.BUY,
                    quantity=2,
                    order_type=OrderType.LIMIT,
                    limit_price=500,
                    reference_price=501,
                    strategy_id="strategy-intent",
                    signal_id="signal-001",
                ),
            ),
        )

        events = account_trade_events_from_proposal_review(
            proposal,
            decision="blocked",
            reviewed_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
            account_id="DU123456",
            review_id="review-001",
            message="Blocked by switch",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "paper_review")
        self.assertEqual(events[0].symbol, "SPY")
        self.assertEqual(events[0].side, "buy")
        self.assertEqual(events[0].price, 501)
        self.assertEqual(events[0].strategy_id, "strategy-intent")
        self.assertEqual(events[0].metadata["decision"], "blocked")
        self.assertEqual(events[0].metadata["review_id"], "review-001")


if __name__ == "__main__":
    unittest.main()
