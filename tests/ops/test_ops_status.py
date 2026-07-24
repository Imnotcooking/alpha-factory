from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from oqp.accounts import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    TradeEvent,
    write_account_snapshot,
    write_account_trade_event,
)
from oqp.brokers import BrokerConnectionStatus
from oqp.config import load_settings
from oqp.ops.status import (
    account_event_items,
    collect_ops_status,
    command_status,
    discord_status_items,
    host_health_items,
    ibkr_api_heartbeat_item,
    latest_account_event_rows,
    latest_account_rows,
    socket_status_item,
)


class OpsStatusTests(unittest.TestCase):
    def test_demo_mode_skips_external_broker_and_scheduler_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = load_settings(root / ".env")
            snapshot = collect_ops_status(
                settings=settings,
                account_ledger_path=root / "accounts.db",
                repo_root=root,
                demo_mode=True,
            )

        names = {item.name for item in snapshot.items}
        self.assertIn("Broker-free demo profile", names)
        self.assertNotIn("Live IBKR Gateway", names)
        self.assertNotIn("IBKR Adapter Heartbeat", names)

    def test_latest_account_rows_redacts_and_sorts_account_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.db"
            write_account_snapshot(
                db_path,
                AccountSnapshot(
                    snapshot_id="acct-live-1",
                    as_of=datetime.now(timezone.utc),
                    account_id="U123456",
                    broker="ibkr",
                    profile="ibkr_live_readonly",
                    environment=AccountEnvironment.LIVE,
                    net_liquidation=40_000,
                    cash=2_000,
                    positions=(
                        PositionSnapshot(
                            symbol="AAPL",
                            asset_class="equity",
                            quantity=1,
                            market_price=200,
                        ),
                    ),
                    cash_balances=(CashSnapshot(currency="USD", cash=2_000),),
                ),
            )
            write_account_snapshot(
                db_path,
                AccountSnapshot(
                    snapshot_id="acct-paper-1",
                    as_of=datetime.now(timezone.utc),
                    account_id="DU123456",
                    broker="ibkr",
                    profile="ibkr_paper_readonly",
                    environment=AccountEnvironment.PAPER,
                    net_liquidation=1_000_000,
                    cash=750_000,
                ),
            )

            rows = latest_account_rows(db_path)

        self.assertEqual([row["environment"] for row in rows], ["live", "paper"])
        self.assertEqual(rows[0]["account_id"], "U1***56")
        self.assertEqual(rows[1]["account_id"], "DU***56")
        self.assertEqual(rows[0]["position_count"], 1)

    def test_latest_account_event_rows_redacts_and_reports_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.db"
            write_account_trade_event(
                db_path,
                TradeEvent(
                    event_id="evt-001",
                    event_type="paper_review",
                    occurred_at=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
                    account_id="DU123456",
                    broker="ibkr",
                    profile="ibkr_paper_readonly",
                    environment=AccountEnvironment.PAPER,
                    symbol="SPY",
                    side="buy",
                    quantity=1,
                    price=500,
                    currency="USD",
                    strategy_id="strategy-001",
                    order_id="proposal-001",
                ),
            )

            rows = latest_account_event_rows(db_path)
            items = account_event_items(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["account_id"], "DU***56")
        self.assertEqual(rows[0]["event_type"], "paper_review")
        self.assertEqual(items[0].status, "pass")
        self.assertIn("events=1", items[0].detail)

    def test_socket_status_item_passes_for_reachable_port(self) -> None:
        with patch("socket.create_connection") as create_connection:
            create_connection.return_value.__enter__.return_value = None
            item = socket_status_item(
                "test socket",
                "127.0.0.1",
                4001,
                timeout=0.5,
            )

        self.assertEqual(item.status, "pass")
        create_connection.assert_called_once()

    def test_ibkr_api_heartbeat_passes_with_connected_readonly_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "IBKR_HOST=127.0.0.1",
                        "IBKR_LIVE_PORT=4001",
                        "IBKR_LIVE_CLIENT_ID=201",
                        "IBKR_LIVE_MONITOR_ENABLED=true",
                        "ALLOW_LIVE_TRADING=false",
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_settings(env_file)

        def fake_fetch(config, *, adapter):
            self.assertEqual(config.client_id, 9243)
            return SimpleNamespace(
                health=SimpleNamespace(
                    status=BrokerConnectionStatus.CONNECTED,
                    account_id="U123456",
                    message="Connected",
                ),
                position_rows=({"Ticker": "AAPL"},),
                metrics={"Available_Cash_USD": 12.34, "Total_NAV_USD": 567.89},
                error=None,
            )

        with patch("oqp.ops.status.os.getpid", return_value=42), patch(
            "oqp.ops.status.get_broker_adapter", return_value=object()
        ), patch(
            "oqp.ops.status.fetch_ibkr_readonly_portfolio_snapshot",
            side_effect=fake_fetch,
        ):
            item = ibkr_api_heartbeat_item(
                "Live IBKR API heartbeat",
                "ibkr_live_readonly",
                settings,
            )

        self.assertEqual(item.status, "pass")
        self.assertIn("Connected read-only", item.detail)
        self.assertEqual(item.metadata["Client ID"], 9243)

    def test_ibkr_api_heartbeat_fails_when_adapter_snapshot_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("IBKR_LIVE_MONITOR_ENABLED=true\n", encoding="utf-8")
            settings = load_settings(env_file)

        with patch("oqp.ops.status.get_broker_adapter", return_value=object()), patch(
            "oqp.ops.status.fetch_ibkr_readonly_portfolio_snapshot",
            return_value=SimpleNamespace(
                health=SimpleNamespace(
                    status=BrokerConnectionStatus.ERROR,
                    account_id=None,
                    message="Could not connect",
                ),
                position_rows=(),
                metrics={},
                error="Could not connect",
            ),
        ):
            item = ibkr_api_heartbeat_item(
                "Live IBKR API heartbeat",
                "ibkr_live_readonly",
                settings,
            )

        self.assertEqual(item.status, "fail")
        self.assertIn("Could not connect", item.detail)

    def test_ibkr_api_heartbeat_warns_when_live_monitor_gate_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("IBKR_LIVE_MONITOR_ENABLED=false\n", encoding="utf-8")
            settings = load_settings(env_file)

        item = ibkr_api_heartbeat_item(
            "Live IBKR API heartbeat",
            "ibkr_live_readonly",
            settings,
        )

        self.assertEqual(item.status, "warn")
        self.assertIn("Skipped", item.detail)

    def test_discord_status_items_detect_configured_webhook(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OQP_DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/abc/def",
            },
            clear=False,
        ):
            items = discord_status_items()

        self.assertEqual(items[0].status, "pass")
        self.assertEqual(items[1].status, "pass")

    def test_command_status_handles_success_and_failure(self) -> None:
        ok = command_status(["python", "-c", "print('ok')"], timeout=5)
        failed = command_status(["python", "-c", "raise SystemExit(7)"], timeout=5)

        self.assertEqual(ok["status"], "pass")
        self.assertIn("ok", ok["stdout"])
        self.assertEqual(failed["status"], "fail")
        self.assertEqual(failed["returncode"], 7)

    def test_host_health_items_warn_on_high_usage(self) -> None:
        items = host_health_items(
            {
                "disk_used_pct": 0.90,
                "disk_free_gb": 1.0,
                "memory_used_pct": 0.95,
                "memory_free_gb": 0.5,
            }
        )

        self.assertEqual(items[0].status, "warn")
        self.assertEqual(items[1].status, "warn")

    def test_snapshot_mode_does_not_warn_on_local_only_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "OQP_OPS_STATUS_SOURCE=snapshot",
                        "ALLOW_LIVE_TRADING=false",
                        "ALLOW_PAPER_TRADING=false",
                        "ALLOW_PAPER_ORDER_SUBMIT=false",
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_settings(env_file)
            root = Path(tmp)
            for name in (
                "portfolio_snapshot_health.json",
                "paper_trading_health.json",
                "ibkr_adapter_heartbeat_health.json",
            ):
                (root / name).write_text(
                    '{"status":"pass","checked_at":"2026-07-01T00:00:00+00:00"}',
                    encoding="utf-8",
                )

            snapshot = collect_ops_status(
                settings=settings,
                account_ledger_path=root / "accounts.db",
                portfolio_health_path=root / "portfolio_snapshot_health.json",
                paper_health_path=root / "paper_trading_health.json",
                ibkr_heartbeat_health_path=root / "ibkr_adapter_heartbeat_health.json",
                repo_root=root,
            )

        rows = snapshot.item_rows
        self.assertTrue(
            any(row["Check"] == "Server-owned scheduling" and row["Status"] == "pass" for row in rows)
        )
        self.assertTrue(
            any(row["Check"] == "Server-side webhooks" and row["Status"] == "pass" for row in rows)
        )
        self.assertTrue(
            any(row["Check"] == "Live monitor evidence" and row["Status"] == "pass" for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
