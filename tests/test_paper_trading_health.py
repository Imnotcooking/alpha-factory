from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from oqp.brokers import (
    BrokerConnectionStatus,
    BrokerHealth,
    IBKRReadOnlyPortfolioSnapshot,
)
from oqp.paper_trading import write_paper_snapshot


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_paper_trading_health.py"


def sample_snapshot() -> IBKRReadOnlyPortfolioSnapshot:
    return IBKRReadOnlyPortfolioSnapshot(
        health=BrokerHealth(
            broker="ibkr",
            status=BrokerConnectionStatus.CONNECTED,
            account_id="DU123456",
        ),
        metrics={
            "Total_NAV_USD": 1_000_000.0,
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


def load_health_module():
    spec = importlib.util.spec_from_file_location("paper_trading_health", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load paper health checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaperTradingHealthTests(unittest.TestCase):
    def test_missing_db_fails(self) -> None:
        health = load_health_module()

        checks, summary = health.run_checks(
            db_path=Path("/tmp/oqp_missing_paper_health.db"),
            max_age_hours=36,
        )

        self.assertEqual(summary, {})
        self.assertTrue(any(check.failed() for check in checks))
        self.assertEqual(checks[0].name, "paper ledger")

    def test_fresh_paper_snapshot_passes(self) -> None:
        health = load_health_module()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "paper.db"
            write_paper_snapshot(db_path, sample_snapshot(), snapshot_date="2026-06-25")
            checks, summary = health.run_checks(
                db_path=db_path,
                max_age_hours=36,
            )

        self.assertFalse(any(check.failed() for check in checks), checks)
        self.assertEqual(summary["account_id"], "DU***56")
        self.assertEqual(summary["position_rows"], 2)
        self.assertEqual(summary["orders_today"], 0)
        self.assertEqual(summary["fills_today"], 0)

    def test_discord_payload_has_content_and_embed(self) -> None:
        health = load_health_module()

        payload = health._discord_payload(
            {
                "status": "pass",
                "checked_at": "2026-06-25T00:00:00+00:00",
                "summary": {
                    "account_id": "DU***56",
                    "net_liquidation": 1_000_000,
                    "cash": 750_000,
                    "daily_pnl": 1234,
                    "position_rows": 2,
                    "orders_today": 0,
                    "fills_today": 0,
                },
                "checks": [],
            }
        )

        self.assertTrue(payload["content"])
        self.assertIn("embeds", payload)
        self.assertEqual(payload["allowed_mentions"], {"parse": []})


if __name__ == "__main__":
    unittest.main()
