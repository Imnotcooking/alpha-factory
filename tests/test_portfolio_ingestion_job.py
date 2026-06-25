from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from oqp.portfolio import (
    load_latest_live_positions,
    run_portfolio_ingestion,
    save_ibkr_metrics,
)


class PortfolioIngestionJobTests(unittest.TestCase):
    def test_saves_ibkr_metrics(self) -> None:
        metrics = {
            "Total_NAV_USD": 100_000.0,
            "Available_Cash_USD": 5_000.0,
            "Margin_Buffer_USD": 25_000.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = save_ibkr_metrics(metrics, metrics_path=Path(tmp) / "metrics.json")
            saved = json.loads(Path(metrics_path).read_text(encoding="utf-8"))

        self.assertEqual(saved, metrics)

    def test_skips_empty_ibkr_metrics_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = save_ibkr_metrics({}, metrics_path=Path(tmp) / "metrics.json")

        self.assertIsNone(metrics_path)

    def test_writes_live_positions_to_runtime_ledger(self) -> None:
        ibkr_positions = pd.DataFrame(
            [
                {
                    "Broker": "IBKR Live",
                    "Ticker": "AAPL",
                    "AssetType": "Equity",
                    "Shares": 3,
                    "AvgPrice": 100.0,
                    "Broker_Price": 110.0,
                    "Broker_PnL": 30.0,
                    "Currency": "USD",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db_path = tmp_path / "portfolio.db"
            state_dir = tmp_path / "state"
            backup_dir = tmp_path / "exports"

            with patch(
                "oqp.portfolio.ingestion_job.fetch_live_ibkr_portfolio",
                return_value=(
                    ibkr_positions,
                    {"Available_Cash_USD": 5_000.0},
                ),
            ):
                result = run_portfolio_ingestion(
                    db_path=db_path,
                    snapshot_date="2026-06-25",
                    raw_dir=tmp_path / "imports",
                    state_dir=state_dir,
                    backup_csv_dir=backup_dir,
                    include_legacy_raw_fallback=False,
                )
            latest = load_latest_live_positions(db_path)
            metrics_exists = bool(result.ibkr_metrics_path and result.ibkr_metrics_path.exists())
            backup_exists = bool(result.backup_csv_path and result.backup_csv_path.exists())

        self.assertEqual(result.status, "updated")
        self.assertEqual(result.position_rows, 1)
        self.assertEqual(result.ibkr_position_rows, 1)
        self.assertTrue(metrics_exists)
        self.assertTrue(backup_exists)
        self.assertEqual(latest.iloc[0]["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
