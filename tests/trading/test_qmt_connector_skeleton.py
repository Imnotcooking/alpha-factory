from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from oqp.accounts import (
    account_snapshot_from_broker_snapshot,
    load_latest_account_positions,
    write_account_snapshot,
)
from oqp.brokers import (
    BrokerConnectionStatus,
    BrokerEnvironment,
    QMTBrokerAdapter,
    get_broker_profile_config,
)
from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, Order, OrderSide, OrderStatus, OrderType


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeQMTSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/health"):
            return FakeResponse(
                {
                    "status": "ok",
                    "connected": True,
                    "account_id": "QMT123456",
                    "mode": "readonly",
                    "mini_qmt_connected": True,
                    "session_id": 880001,
                }
            )
        if url.endswith("/account"):
            return FakeResponse(
                {
                    "account": {
                        "account_id": "QMT123456",
                        "account_type": "STOCK",
                        "currency": "CNY",
                        "cash": 800000,
                        "market_value": 200000,
                        "total_asset": 1000000,
                    }
                }
            )
        if url.endswith("/positions"):
            return FakeResponse(
                {
                    "positions": [
                        {
                            "symbol": "600000.SH",
                            "asset_class": "equity",
                            "quantity": 1000,
                            "avg_price": 10.0,
                            "market_price": 10.5,
                            "market_value": 10500,
                            "unrealized_pnl": 500,
                            "currency": "CNY",
                            "multiplier": 1,
                        },
                        {
                            "symbol": "rb2401.SF",
                            "asset_class": "future",
                            "position": 2,
                            "direction": "short",
                            "avg_price": 3600,
                            "last_price": 3580,
                            "instrument_value": 71600,
                            "float_profit": 400,
                            "currency": "CNY",
                            "multiplier": 10,
                        },
                    ]
                }
            )
        if url.endswith("/orders"):
            return FakeResponse({"orders": []})
        if url.endswith("/submit_order"):
            return FakeResponse(
                {
                    "order": {
                        "order_id": "9001",
                        "order_status": 50,
                        "message": "submitted",
                    }
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")


class QMTConnectorSkeletonTests(unittest.TestCase):
    def test_qmt_profiles_are_locked_by_default(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "QMT_CONNECTOR_URL=http://10.0.0.8:58668",
                "QMT_PAPER_ACCOUNT_ID=PAPER123",
                "QMT_ACCOUNT_TYPE=FUTURE",
            ]
        )
        self.addCleanup(tmp.cleanup)

        readonly = get_broker_profile_config("qmt_paper_readonly", settings=settings)
        self.assertEqual(readonly.broker, "qmt")
        self.assertEqual(readonly.host, "10.0.0.8")
        self.assertEqual(readonly.port, 58668)
        self.assertEqual(readonly.account_id, "PAPER123")
        self.assertEqual(readonly.metadata["account_type"], "FUTURE")

        with self.assertRaises(ValueError):
            get_broker_profile_config("qmt_paper_submit", settings=settings)

    def test_qmt_paper_submit_requires_global_and_qmt_switches(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "QMT_CONNECTOR_URL=http://127.0.0.1:58668",
                "QMT_SUBMIT_CONNECTOR_URL=http://127.0.0.1:58669",
                "QMT_API_TOKEN=secret",
                "QMT_REQUEST_SIGNING_SECRET=signing-secret",
                "ALLOW_LIVE_TRADING=false",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_QMT_PAPER_ORDER_SUBMIT=true",
                "QMT_PAPER_ACCOUNT_ID=PAPER123",
            ]
        )
        self.addCleanup(tmp.cleanup)

        config = get_broker_profile_config("qmt_paper_submit", settings=settings)
        self.assertEqual(config.environment, BrokerEnvironment.PAPER)
        self.assertFalse(config.readonly)
        self.assertEqual(config.metadata["profile"], "qmt_paper_submit")
        self.assertEqual(config.metadata["connector_url"], "http://127.0.0.1:58669")

    def test_qmt_paper_submit_requires_token_signing_and_isolated_url(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "QMT_CONNECTOR_URL=http://127.0.0.1:58668",
                "QMT_SUBMIT_CONNECTOR_URL=http://127.0.0.1:58668",
                "ALLOW_LIVE_TRADING=false",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_QMT_PAPER_ORDER_SUBMIT=true",
                "QMT_PAPER_ACCOUNT_ID=PAPER123",
            ]
        )
        self.addCleanup(tmp.cleanup)

        with self.assertRaises(ValueError):
            get_broker_profile_config("qmt_paper_submit", settings=settings)

    def test_adapter_parses_fake_connector_snapshot(self) -> None:
        tmp, settings = settings_from_lines(["QMT_PAPER_ACCOUNT_ID=QMT123456"])
        self.addCleanup(tmp.cleanup)
        config = get_broker_profile_config("qmt_paper_readonly", settings=settings)
        adapter = QMTBrokerAdapter(session=FakeQMTSession())

        health = adapter.connect(config)
        snapshot = adapter.get_snapshot()

        self.assertEqual(health.status, BrokerConnectionStatus.CONNECTED)
        self.assertEqual(snapshot.account.currency, "CNY")
        self.assertEqual(snapshot.account.net_liquidation, 1000000)
        self.assertEqual(len(snapshot.positions), 2)
        self.assertEqual(snapshot.positions[1].instrument.asset_class, AssetClass.FUTURE)
        self.assertEqual(snapshot.positions[1].quantity, -2)

    def test_qmt_snapshot_writes_to_account_ledger(self) -> None:
        tmp, settings = settings_from_lines(["QMT_PAPER_ACCOUNT_ID=QMT123456"])
        self.addCleanup(tmp.cleanup)
        config = get_broker_profile_config("qmt_paper_readonly", settings=settings)
        adapter = QMTBrokerAdapter(session=FakeQMTSession())
        adapter.connect(config)
        snapshot = adapter.get_snapshot()

        with tempfile.TemporaryDirectory() as db_tmp:
            db_path = Path(db_tmp) / "accounts.db"
            account_snapshot = account_snapshot_from_broker_snapshot(
                snapshot,
                environment="paper",
                profile="qmt_paper_readonly",
                snapshot_date="2026-07-07",
            )
            result = write_account_snapshot(
                db_path,
                account_snapshot,
                snapshot_date="2026-07-07",
            )
            positions = load_latest_account_positions(db_path, environment="paper")

        self.assertEqual(result.position_rows, 2)
        self.assertEqual(positions.iloc[0]["broker"], "qmt")
        self.assertIn("600000.SH", set(positions["symbol"]))

    def test_submit_order_posts_to_connector_when_profile_is_write_enabled(self) -> None:
        tmp, settings = settings_from_lines(
            [
                "QMT_CONNECTOR_URL=http://127.0.0.1:58668",
                "QMT_SUBMIT_CONNECTOR_URL=http://127.0.0.1:58669",
                "QMT_API_TOKEN=secret",
                "QMT_REQUEST_SIGNING_SECRET=signing-secret",
                "ALLOW_LIVE_TRADING=false",
                "ALLOW_PAPER_ORDER_SUBMIT=true",
                "ALLOW_QMT_PAPER_ORDER_SUBMIT=true",
                "QMT_PAPER_ACCOUNT_ID=QMT123456",
            ]
        )
        self.addCleanup(tmp.cleanup)
        session = FakeQMTSession()
        config = get_broker_profile_config("qmt_paper_submit", settings=settings)
        adapter = QMTBrokerAdapter(session=session)
        adapter.connect(config)

        receipt = adapter.place_order(
            Order(
                instrument=Instrument(
                    symbol="600000.SH",
                    asset_class=AssetClass.EQUITY,
                    currency="CNY",
                    multiplier=1,
                ),
                side=OrderSide.BUY,
                quantity=100,
                order_type=OrderType.LIMIT,
                limit_price=10.5,
                strategy_id="qmt_demo",
                client_order_id="paper-qmt-demo-1",
            )
        )

        submit_call = [call for call in session.calls if call["url"].endswith("/submit_order")][0]
        submit_payload = submit_call.get("json")
        if submit_payload is None:
            submit_payload = json.loads(submit_call["data"].decode("utf-8"))
        self.assertEqual(submit_payload["account_id"], "QMT123456")
        self.assertEqual(submit_payload["symbol"], "600000.SH")
        self.assertIn("X-OQP-Signature", submit_call["headers"])
        self.assertEqual(receipt.status, OrderStatus.SUBMITTED)
        self.assertEqual(receipt.broker_order_id, "9001")


if __name__ == "__main__":
    unittest.main()
