from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from oqp.portfolio.ingestion_job import save_ibkr_metrics


class MiddleOfficeETLRetirementTests(unittest.TestCase):
    def test_legacy_metrics_contract_is_covered_by_canonical_ingestion(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
