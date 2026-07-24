from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from oqp.investing import (
    load_opportunity_history,
    load_opportunity_snapshot_detail,
    write_opportunity_snapshot,
)


class OpportunityHistoryTests(unittest.TestCase):
    def test_write_and_load_opportunity_snapshot(self) -> None:
        lens = pd.DataFrame([{"Lens": "Valuation", "Score": 0.8, "Bias": "constructive"}])
        route = pd.DataFrame([{"Vehicle": "Defined-risk spread", "Fit": "High"}])
        playbook = pd.DataFrame([{"Theme": "Defined-risk spread", "Fit": "High", "Best Candidate": "Bull Call"}])
        catalyst = pd.DataFrame([{"Catalyst": "Earnings", "Articles": 2, "Tone": "Positive"}])

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "opportunity_history.db"
            summary = write_opportunity_snapshot(
                symbol="aapl",
                spot=100.0,
                action_bucket="Defined-risk spread",
                primary_route="Defined-risk spread",
                direction="Bullish",
                direction_score=0.45,
                news_tone="Positive",
                news_score=0.22,
                target_upside=0.18,
                forecast_vol=0.35,
                market_iv=0.28,
                reference_expiry="2026-07-31",
                lens_frame=lens,
                route_frame=route,
                playbook_frame=playbook,
                catalyst_frame=catalyst,
                thesis_metadata={"conviction": "medium"},
                path=db_path,
                captured_at=datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc),
            )
            history = load_opportunity_history("AAPL", path=db_path)
            detail = load_opportunity_snapshot_detail(summary["snapshot_id"], path=db_path)

        self.assertEqual(summary["symbol"], "AAPL")
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["primary_route"], "Defined-risk spread")
        self.assertEqual(detail["playbook"][0]["Best Candidate"], "Bull Call")
        self.assertEqual(detail["thesis"]["conviction"], "medium")

    def test_load_empty_history_has_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            frame = load_opportunity_history("MSFT", path=Path(tmp) / "empty.db")

        self.assertIn("snapshot_id", frame.columns)
        self.assertTrue(frame.empty)


if __name__ == "__main__":
    unittest.main()
