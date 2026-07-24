from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.brokers import (  # noqa: E402
    AccountSummary,
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    BrokerEnvironment,
    BrokerHealth,
    IBKRBrokerAdapter,
    fetch_ibkr_readonly_portfolio_snapshot,
    ibkr_account_summary_to_live_metrics,
    ibkr_position_to_live_position_row,
)
from oqp.domain import AssetClass, Instrument, Position  # noqa: E402


class FakeIBKRAdapter:
    def __init__(self, *, connected: bool = True, raise_on_positions: bool = False) -> None:
        self.connected = connected
        self.raise_on_positions = raise_on_positions
        self.disconnected = False

    def connect(self, config: BrokerConnectionConfig) -> BrokerHealth:
        status = (
            BrokerConnectionStatus.CONNECTED
            if self.connected
            else BrokerConnectionStatus.ERROR
        )
        return BrokerHealth(
            broker="ibkr",
            status=status,
            account_id="DU123",
            message=None if self.connected else "offline",
        )

    def disconnect(self) -> None:
        self.disconnected = True

    def get_account_summary(self) -> AccountSummary:
        return AccountSummary(
            broker="ibkr",
            account_id="DU123",
            currency="USD",
            net_liquidation=100_000.0,
            cash=5_000.0,
            buying_power=50_000.0,
            metadata={"excess_liquidity": 12_500.0},
        )

    def get_positions(self) -> list[Position]:
        if self.raise_on_positions:
            raise RuntimeError("portfolio fetch failed")
        return [
            Position(
                instrument=Instrument(
                    symbol="AAPL",
                    asset_class=AssetClass.EQUITY,
                    currency="USD",
                    broker_symbol="AAPL",
                ),
                quantity=10.0,
                average_cost=100.0,
                market_price=None,
                broker="ibkr",
                metadata={"unrealized_pnl": 25.0},
            ),
            Position(
                instrument=Instrument(
                    symbol="AAPL 260116C00150000",
                    asset_class=AssetClass.OPTION,
                    currency="USD",
                    broker_symbol="AAPL260116C150000",
                    multiplier=100.0,
                ),
                quantity=2.0,
                average_cost=1.25,
                market_price=1.5,
                broker="ibkr",
                metadata={"unrealized_pnl": 50.0},
            ),
        ]


class FakeSummaryItem:
    def __init__(self, tag: str, value: str, currency: str, account: str = "U123") -> None:
        self.tag = tag
        self.value = value
        self.currency = currency
        self.account = account


class FakeConnectedIB:
    def isConnected(self) -> bool:
        return True

    def managedAccounts(self) -> list[str]:
        return ["U123"]

    def accountSummary(self) -> list[FakeSummaryItem]:
        return [
            FakeSummaryItem("BaseCurrency", "EUR", ""),
            FakeSummaryItem("NetLiquidation", "38025.43", "BASE"),
            FakeSummaryItem("TotalCashValue", "0.0", "BASE"),
            FakeSummaryItem("TotalCashValue", "563.52", "EUR"),
            FakeSummaryItem("AvailableFunds", "1200.00", "BASE"),
            FakeSummaryItem("GrossPositionValue", "41000.00", "BASE"),
        ]


class IBKRReadOnlyPortfolioTests(unittest.TestCase):
    def test_converts_position_to_live_position_row(self) -> None:
        position = Position(
            instrument=Instrument(
                symbol="MSFT",
                asset_class=AssetClass.EQUITY,
                currency="USD",
                broker_symbol="MSFT",
            ),
            quantity=3.0,
            average_cost=200.0,
            market_price=None,
            metadata={"unrealized_pnl": None},
        )

        row = ibkr_position_to_live_position_row(position)

        self.assertEqual(row["Ticker"], "MSFT")
        self.assertEqual(row["Shares"], 3.0)
        self.assertEqual(row["AvgPrice"], 200.0)
        self.assertEqual(row["Broker_Price"], 200.0)
        self.assertEqual(row["Broker_PnL"], 0.0)
        self.assertEqual(row["AssetType"], "Equity")
        self.assertEqual(row["Broker"], "IBKR Live")

    def test_converts_account_summary_to_live_metrics(self) -> None:
        metrics = ibkr_account_summary_to_live_metrics(
            AccountSummary(
                broker="ibkr",
                account_id="DU123",
                currency="USD",
                net_liquidation=100_000.0,
                cash=5_000.0,
                buying_power=50_000.0,
                metadata={"excess_liquidity": 12_500.0},
            )
        )

        self.assertEqual(metrics["Account_Currency"], "USD")
        self.assertEqual(metrics["Total_NAV"], 100_000.0)
        self.assertEqual(metrics["Available_Cash"], 5_000.0)
        self.assertEqual(metrics["Margin_Buffer"], 12_500.0)
        self.assertEqual(metrics["Total_NAV_USD"], 100_000.0)
        self.assertEqual(metrics["Available_Cash_USD"], 5_000.0)
        self.assertEqual(metrics["Margin_Buffer_USD"], 12_500.0)

    def test_account_summary_falls_back_to_account_currency_cash(self) -> None:
        adapter = IBKRBrokerAdapter()
        adapter._ib = FakeConnectedIB()

        summary = adapter.get_account_summary()

        self.assertEqual(summary.currency, "EUR")
        self.assertEqual(summary.net_liquidation, 38_025.43)
        self.assertEqual(summary.cash, 563.52)

    def test_fetches_readonly_snapshot_and_disconnects(self) -> None:
        adapter = FakeIBKRAdapter()
        config = BrokerConnectionConfig(
            broker="ibkr",
            host="127.0.0.1",
            port=7496,
            client_id=201,
            environment=BrokerEnvironment.LIVE,
            readonly=True,
        )

        snapshot = fetch_ibkr_readonly_portfolio_snapshot(config, adapter=adapter)

        self.assertIsNone(snapshot.error)
        self.assertTrue(adapter.disconnected)
        self.assertEqual(snapshot.metrics["Total_NAV_USD"], 100_000.0)
        self.assertEqual(len(snapshot.position_rows), 2)
        self.assertEqual(snapshot.position_rows[0]["Broker_Price"], 100.0)
        self.assertEqual(snapshot.position_rows[1]["AssetType"], "Option")
        self.assertEqual(snapshot.position_rows[1]["Multiplier"], 100.0)

    def test_returns_error_snapshot_when_not_connected(self) -> None:
        adapter = FakeIBKRAdapter(connected=False)
        config = BrokerConnectionConfig(
            broker="ibkr",
            host="127.0.0.1",
            port=7496,
            client_id=201,
            environment=BrokerEnvironment.LIVE,
            readonly=True,
        )

        snapshot = fetch_ibkr_readonly_portfolio_snapshot(config, adapter=adapter)

        self.assertEqual(snapshot.health.status, BrokerConnectionStatus.ERROR)
        self.assertEqual(snapshot.error, "offline")
        self.assertEqual(snapshot.position_rows, ())
        self.assertEqual(snapshot.metrics, {})


if __name__ == "__main__":
    unittest.main()
