from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from oqp.portfolio import write_historical_nav, write_live_positions_frame


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_portfolio_snapshot_health.py"


def load_health_module():
    spec = importlib.util.spec_from_file_location("portfolio_snapshot_health", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load health checker script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PortfolioSnapshotHealthTests(unittest.TestCase):
    def test_missing_db_fails(self) -> None:
        health = load_health_module()

        with tempfile.TemporaryDirectory() as tmp:
            checks = health.run_checks(
                db_path=Path(tmp) / "missing.db",
                ibkr_metrics_path=Path(tmp) / "missing_metrics.json",
                max_age_hours=36,
            )

        self.assertTrue(any(check.failed() for check in checks))
        self.assertEqual(checks[0].name, "portfolio ledger")
        self.assertEqual(checks[0].status, "fail")

    def test_fresh_nav_and_positions_pass(self) -> None:
        health = load_health_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "portfolio.db"
            metrics_path = root / "ibkr_metrics.json"
            metrics_path.write_text('{"Total_NAV_USD": 12345.0}\n')
            write_live_positions_frame(
                db_path,
                pd.DataFrame(
                    [
                        {
                            "Broker": "IBKR Live",
                            "Ticker": "AAPL",
                            "AssetType": "Equity",
                            "Shares": 2,
                            "AvgPrice": 100,
                            "Broker_Price": 110,
                            "Broker_PnL": 20,
                            "Currency": "USD",
                        }
                    ]
                ),
                snapshot_date=date.today(),
            )
            write_historical_nav(
                db_path,
                snapshot_date=date.today(),
                total_net_worth=12345.0,
                total_cash=1000.0,
                portfolio_beta=1.1,
            )

            checks = health.run_checks(
                db_path=db_path,
                ibkr_metrics_path=metrics_path,
                max_age_hours=36,
            )

        self.assertFalse(any(check.failed() for check in checks), checks)
        self.assertIn("latest NAV", {check.name for check in checks})
        self.assertIn("latest positions", {check.name for check in checks})

    def test_stale_nav_fails(self) -> None:
        health = load_health_module()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "portfolio.db"
            write_historical_nav(
                db_path,
                snapshot_date="2020-01-01",
                total_net_worth=12345.0,
            )
            checks = health.run_checks(
                db_path=db_path,
                ibkr_metrics_path=Path(tmp) / "missing_metrics.json",
                max_age_hours=1,
            )

        freshness = [check for check in checks if check.name == "NAV freshness"][0]
        self.assertEqual(freshness.status, "fail")

    def test_discord_payload_has_message_content(self) -> None:
        health = load_health_module()

        payload = health._discord_payload(
            {
                "status": "pass",
                "checked_at": "2026-06-25T00:00:00+00:00",
                "checks": [
                    {
                        "name": "latest NAV",
                        "status": "pass",
                        "detail": "date=2026-06-25 total_net_worth=41,573.45",
                    },
                    {
                        "name": "NAV freshness",
                        "status": "pass",
                        "detail": "Latest NAV is 0.7 hours old.",
                    },
                ],
            }
        )

        self.assertIn("content", payload)
        self.assertIn("embeds", payload)
        self.assertEqual(payload["allowed_mentions"], {"parse": []})
        self.assertTrue(payload["content"])


if __name__ == "__main__":
    unittest.main()
