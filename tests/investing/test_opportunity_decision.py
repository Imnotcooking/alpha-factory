from __future__ import annotations

import unittest

import pandas as pd

from oqp.investing import (
    build_decision_checklist_frame,
    build_options_playbook_frame,
    build_opportunity_lens_frame,
    build_thesis_draft,
    build_vehicle_route_frame,
    primary_route,
)


class OpportunityDecisionTests(unittest.TestCase):
    def test_lens_frame_scores_core_inputs(self) -> None:
        frame = build_opportunity_lens_frame(
            spot=100.0,
            target_consensus=125.0,
            direction="Bullish",
            direction_score=0.45,
            news_label="Slightly Positive",
            news_score=0.12,
            forecast_vol=0.35,
            market_iv=0.25,
            rsi_14=38.0,
            expiration_count=12,
        )

        self.assertEqual(list(frame.columns), ["Lens", "Score", "Bias", "Detail"])
        rows = frame.set_index("Lens")
        self.assertEqual(rows.loc["Valuation", "Bias"], "constructive")
        self.assertGreater(rows.loc["Vol Pricing", "Score"], 0)
        self.assertEqual(rows.loc["Timing", "Bias"], "neutral")

    def test_route_frame_favors_shares_when_upside_is_constructive(self) -> None:
        frame = build_vehicle_route_frame(
            target_upside=0.22,
            direction="Bullish",
            direction_score=0.32,
            news_score=0.08,
            forecast_vol=0.30,
            market_iv=0.27,
        )

        rows = frame.set_index("Vehicle")
        self.assertEqual(rows.loc["Shares / staged entry", "Fit"], "High")
        self.assertIn(rows.loc["Defined-risk spread", "Fit"], {"High", "Medium"})

    def test_route_frame_favors_income_when_iv_is_rich_and_direction_neutral(self) -> None:
        frame = build_vehicle_route_frame(
            target_upside=0.03,
            direction="Neutral",
            direction_score=0.05,
            news_score=0.0,
            forecast_vol=0.25,
            market_iv=0.35,
        )

        rows = frame.set_index("Vehicle")
        self.assertEqual(rows.loc["Income / short vol", "Fit"], "High")
        self.assertEqual(rows.loc["Long options", "Fit"], "Low")

    def test_options_playbook_combines_route_fit_and_candidates(self) -> None:
        route_frame = pd.DataFrame(
            [
                {"Vehicle": "Long options", "Fit": "Medium"},
                {"Vehicle": "Defined-risk spread", "Fit": "High"},
                {"Vehicle": "Income / short vol", "Fit": "Low"},
            ]
        )
        candidates = pd.DataFrame(
            [
                {"Strategy": "Long Call", "Structure": "+1x 100C", "PoP": 0.42, "EV": 120.0, "Max Loss": -450.0},
                {"Strategy": "Bull Call Spread", "Structure": "+1x 95C / -1x 105C", "PoP": 0.55, "EV": 180.0, "Max Loss": -300.0},
                {"Strategy": "Iron Condor", "Structure": "90P/95P/110C/115C", "PoP": 0.68, "EV": 70.0, "Max Loss": -500.0},
            ]
        )

        frame = build_options_playbook_frame(
            route_frame=route_frame,
            candidates=candidates,
            forecast_vol=0.35,
            market_iv=0.28,
            reference_expiry="2026-07-31",
            chain_source="Massive",
        )

        self.assertEqual(frame.iloc[0]["Theme"], "Defined-risk spread")
        row = frame.set_index("Theme").loc["Defined-risk spread"]
        self.assertEqual(row["Fit"], "High")
        self.assertIn("Bull Call Spread", row["Best Candidate"])
        self.assertEqual(row["Source"], "Massive")

    def test_thesis_draft_and_checklist_use_primary_route(self) -> None:
        route_frame = pd.DataFrame(
            [
                {"Vehicle": "Watch / wait", "Fit": "Medium"},
                {"Vehicle": "Defined-risk spread", "Fit": "High"},
            ]
        )

        draft = build_thesis_draft(
            symbol="AAPL",
            action_bucket="Defined-risk spread",
            action_reason="Direction is constructive but premium should be controlled.",
            route_frame=route_frame,
            direction="Bullish",
            direction_score=0.42,
            news_label="Slightly Positive",
            news_score=0.11,
            top_topics="Earnings, Analyst Action",
            top_keywords="growth, upgrade",
        )
        checklist = build_decision_checklist_frame(
            route_frame=route_frame,
            article_count=8,
            target_upside=0.18,
            expiration_count=12,
            market_iv=0.28,
            forecast_vol=0.34,
        )

        self.assertEqual(primary_route(route_frame), "Defined-risk spread")
        self.assertIn("AAPL thesis draft", draft)
        self.assertIn("Preferred route: Defined-risk spread", draft)
        self.assertEqual(checklist.iloc[0]["Status"], "ready")
        self.assertEqual(checklist.loc[checklist["Check"].eq("News context loaded"), "Status"].iloc[0], "ready")


if __name__ == "__main__":
    unittest.main()
