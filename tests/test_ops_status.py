from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from oqp.accounts import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    write_account_snapshot,
)
from oqp.ops.status import (
    command_status,
    discord_status_items,
    host_health_items,
    latest_account_rows,
    socket_status_item,
)


class OpsStatusTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
