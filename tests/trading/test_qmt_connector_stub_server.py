from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from oqp.brokers import (
    BrokerConnectionStatus,
    QMTBrokerAdapter,
    get_broker_profile_config,
)
from oqp.config import load_settings
from oqp.domain import AssetClass, Instrument, Order, OrderSide, OrderStatus, OrderType
from oqp.qmt_connector import (
    ConnectorRiskPolicy,
    PAPER_SUBMIT_MODE,
    READONLY_MODE,
    StubConnectorState,
    create_qmt_connector_stub_server,
    start_qmt_connector_stub,
)
from oqp.qmt_connector.security import qmt_json_body_bytes, qmt_signature_headers


def settings_from_lines(lines: list[str]):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    settings = load_settings(path)
    return tmp, settings


def request_json(
    url: str,
    *,
    token: str | None = None,
    signing_secret: str | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
    body: dict | None = None,
) -> dict:
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    parsed_url = urlparse(url)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = qmt_json_body_bytes(body)
        headers["Content-Type"] = "application/json"
        method = "POST"
    if signing_secret:
        headers.update(
            qmt_signature_headers(
                signing_secret,
                method,
                parsed_url.path,
                params=parsed_url.query,
                body=data or b"",
                timestamp=timestamp,
                nonce=nonce,
            )
        )
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def valid_policy() -> ConnectorRiskPolicy:
    return ConnectorRiskPolicy(
        allowed_symbols=("600000.SH",),
        allowed_asset_classes=("equity",),
        allowed_account_types=("STOCK",),
        max_quantity=1000,
        max_notional=20_000,
        min_limit_price=1,
        max_limit_price=20,
    )


class QMTConnectorStubServerTests(unittest.TestCase):
    def test_stub_health_requires_token_when_configured(self) -> None:
        running = start_qmt_connector_stub(
            state=StubConnectorState(mode=READONLY_MODE),
            api_token="secret",
        )
        self.addCleanup(running.stop)

        with self.assertRaises(HTTPError) as ctx:
            request_json(f"{running.base_url}/health")
        self.assertEqual(ctx.exception.code, 401)

        payload = request_json(f"{running.base_url}/health", token="secret")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["mode"], READONLY_MODE)

    def test_submit_mode_requires_token_signing_private_bind_and_risk_policy(self) -> None:
        with self.assertRaises(ValueError):
            create_qmt_connector_stub_server(
                state=StubConnectorState(mode=PAPER_SUBMIT_MODE),
                api_token="secret",
                signing_secret="signing-secret",
            )
        with self.assertRaises(ValueError):
            create_qmt_connector_stub_server(
                state=StubConnectorState(mode=PAPER_SUBMIT_MODE, risk_policy=valid_policy()),
                signing_secret="signing-secret",
            )
        with self.assertRaises(ValueError):
            create_qmt_connector_stub_server(
                state=StubConnectorState(mode=PAPER_SUBMIT_MODE, risk_policy=valid_policy()),
                api_token="secret",
            )
        with self.assertRaises(ValueError):
            create_qmt_connector_stub_server("0.0.0.0", state=StubConnectorState())

    def test_adapter_reads_account_positions_from_stub(self) -> None:
        running = start_qmt_connector_stub(
            state=StubConnectorState(account_id="PAPER123", mode=READONLY_MODE),
            api_token="secret",
        )
        self.addCleanup(running.stop)
        tmp, settings = settings_from_lines(
            [
                f"QMT_CONNECTOR_URL={running.base_url}",
                "QMT_API_TOKEN=secret",
                "QMT_PAPER_ACCOUNT_ID=PAPER123",
            ]
        )
        self.addCleanup(tmp.cleanup)

        config = get_broker_profile_config("qmt_paper_readonly", settings=settings)
        adapter = QMTBrokerAdapter()
        health = adapter.connect(config)
        snapshot = adapter.get_snapshot()

        self.assertEqual(health.status, BrokerConnectionStatus.CONNECTED)
        self.assertEqual(snapshot.account.account_id, "PAPER123")
        self.assertEqual(snapshot.account.currency, "CNY")
        self.assertEqual(len(snapshot.positions), 2)
        self.assertEqual(snapshot.positions[1].instrument.asset_class, AssetClass.FUTURE)
        self.assertLess(snapshot.positions[1].quantity, 0)

    def test_readonly_stub_rejects_submit_even_when_client_is_write_enabled(self) -> None:
        running = start_qmt_connector_stub(state=StubConnectorState(mode=READONLY_MODE))
        self.addCleanup(running.stop)

        with self.assertRaises(HTTPError) as ctx:
            request_json(
                f"{running.base_url}/submit_order",
                body={
                    "account_id": "PAPER123",
                    "symbol": "600000.SH",
                    "side": "buy",
                    "quantity": 100,
                    "order_type": "limit",
                    "limit_price": 10.5,
                },
            )
        self.assertEqual(ctx.exception.code, 403)

    def test_paper_submit_adapter_places_and_cancels_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            connector_audit_path = Path(tmpdir) / "windows_qmt_audit.jsonl"
            oqp_audit_path = Path(tmpdir) / "oqp_qmt_audit.jsonl"
            running = start_qmt_connector_stub(
                state=StubConnectorState(
                    account_id="PAPER123",
                    mode=PAPER_SUBMIT_MODE,
                    risk_policy=valid_policy(),
                    audit_log_path=connector_audit_path,
                ),
                api_token="secret",
                signing_secret="signing-secret",
            )
            self.addCleanup(running.stop)
            env_tmp, settings = settings_from_lines(
                [
                    "QMT_CONNECTOR_URL=http://127.0.0.1:58668",
                    f"QMT_SUBMIT_CONNECTOR_URL={running.base_url}",
                    "QMT_API_TOKEN=secret",
                    "QMT_REQUEST_SIGNING_SECRET=signing-secret",
                    f"QMT_AUDIT_LOG_PATH={oqp_audit_path}",
                    "QMT_PAPER_ACCOUNT_ID=PAPER123",
                    "ALLOW_LIVE_TRADING=false",
                    "ALLOW_PAPER_ORDER_SUBMIT=true",
                    "ALLOW_QMT_PAPER_ORDER_SUBMIT=true",
                ]
            )
            self.addCleanup(env_tmp.cleanup)

            config = get_broker_profile_config("qmt_paper_submit", settings=settings)
            adapter = QMTBrokerAdapter()
            adapter.connect(config)

            order = Order(
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
                strategy_id="qmt_stub_test",
                client_order_id="stub-test-1",
            )
            receipt = adapter.place_order(order)
            replay_receipt = adapter.place_order(order)

            self.assertEqual(receipt.status, OrderStatus.SUBMITTED)
            self.assertEqual(receipt.broker_order_id, "9001")
            self.assertEqual(replay_receipt.broker_order_id, "9001")
            self.assertTrue(replay_receipt.metadata["raw"]["idempotent_replay"])
            self.assertEqual(len(running.state.orders), 1)
            self.assertEqual(len(adapter.get_open_orders()), 1)

            cancel = adapter.cancel_order("9001")

            self.assertTrue(cancel.cancelled)
            self.assertEqual(cancel.status, OrderStatus.CANCELLED)
            self.assertEqual(adapter.get_open_orders(), ())

            connector_audit = connector_audit_path.read_text(encoding="utf-8").splitlines()
            oqp_audit = oqp_audit_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(connector_audit), 3)
            self.assertGreaterEqual(len(oqp_audit), 3)
            self.assertIn("connector_submit_order", connector_audit[0])
            self.assertIn("oqp_submit_order", oqp_audit[0])

    def test_submit_risk_limits_and_signature_replay_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "windows_qmt_audit.jsonl"
            running = start_qmt_connector_stub(
                state=StubConnectorState(
                    account_id="PAPER123",
                    mode=PAPER_SUBMIT_MODE,
                    risk_policy=valid_policy(),
                    audit_log_path=audit_path,
                ),
                api_token="secret",
                signing_secret="signing-secret",
            )
            self.addCleanup(running.stop)

            bad_symbol = {
                "account_id": "PAPER123",
                "account_type": "STOCK",
                "symbol": "000001.SZ",
                "asset_class": "equity",
                "side": "buy",
                "quantity": 100,
                "order_type": "limit",
                "limit_price": 10.5,
                "client_order_id": "risk-reject-1",
            }
            with self.assertRaises(HTTPError) as risk_ctx:
                request_json(
                    f"{running.base_url}/submit_order",
                    token="secret",
                    signing_secret="signing-secret",
                    body=bad_symbol,
                )
            self.assertEqual(risk_ctx.exception.code, 403)

            good_order = {
                "account_id": "PAPER123",
                "account_type": "STOCK",
                "symbol": "600000.SH",
                "asset_class": "equity",
                "side": "buy",
                "quantity": 100,
                "order_type": "limit",
                "limit_price": 10.5,
                "client_order_id": "replay-test-1",
            }
            request_json(
                f"{running.base_url}/submit_order",
                token="secret",
                signing_secret="signing-secret",
                timestamp=int(time.time()),
                nonce="fixed-replay-nonce",
                body=good_order,
            )
            with self.assertRaises(HTTPError) as replay_ctx:
                request_json(
                    f"{running.base_url}/submit_order",
                    token="secret",
                    signing_secret="signing-secret",
                    timestamp=int(time.time()),
                    nonce="fixed-replay-nonce",
                    body=good_order,
                )
            self.assertEqual(replay_ctx.exception.code, 401)
            audit = audit_path.read_text(encoding="utf-8")
            self.assertIn("risk check failed", audit)
            self.assertIn("replayed signature nonce", audit)


if __name__ == "__main__":
    unittest.main()
