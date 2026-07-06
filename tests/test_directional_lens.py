from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oqp.intelligence.signal_engine import (
    add_strategy_direction_columns,
    build_directional_lens,
    strategy_alignment,
    strategy_payoff_direction,
)


class DirectionalLensTests(unittest.TestCase):
    def sample_history(self, rows: int = 90) -> pd.DataFrame:
        dates = pd.bdate_range("2026-02-01", periods=rows)
        close = np.linspace(100, 124, rows) + np.sin(np.arange(rows) / 3) * 1.5
        return pd.DataFrame({"Close": close}, index=dates)

    def option_chain(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls = pd.DataFrame(
            {
                "strike": [115, 120, 125, 130],
                "impliedVolatility": [0.31, 0.30, 0.34, 0.36],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [105, 110, 115, 120],
                "impliedVolatility": [0.29, 0.30, 0.31, 0.32],
            }
        )
        return calls, puts

    def test_build_directional_lens_returns_horizons_and_contributions(self) -> None:
        calls, puts = self.option_chain()
        lens = build_directional_lens(
            "aapl",
            self.sample_history(),
            spot=124,
            price_targets={"targetConsensus": 150, "recommendationKey": "buy"},
            sentiment_payload={
                "news": [
                    {"publishedDate": "2026-06-29", "title": "AAPL demand improves", "sentiment": "Positive"},
                    {"publishedDate": "2026-06-28", "title": "Neutral note", "sentiment": "Neutral"},
                ]
            },
            calls=calls,
            puts=puts,
        )

        self.assertEqual(lens.symbol, "AAPL")
        self.assertEqual(set(lens.horizon_frame["Horizon"]), {"1D", "1W", "1M"})
        self.assertFalse(lens.contribution_frame.empty)
        self.assertIn(lens.summary["primary_direction"], {"Bullish", "Bearish", "Neutral"})

    def test_strategy_direction_and_alignment(self) -> None:
        self.assertEqual(strategy_payoff_direction("Bull Call Spread"), "Bullish")
        self.assertEqual(strategy_alignment("Bullish", "Bullish", 0.4), "Agree")
        self.assertEqual(strategy_alignment("Bearish", "Bullish", 0.4), "Conflict")

    def test_add_strategy_direction_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"Strategy": "Bull Call Spread", "EV": 100.0},
                {"Strategy": "Bear Put Spread", "EV": 50.0},
            ]
        )
        horizon = pd.DataFrame([{"Horizon": "1W", "Direction": "Bullish", "Score": 0.45}])
        enriched = add_strategy_direction_columns(frame, horizon, horizon="1W")

        self.assertEqual(enriched.iloc[0]["Payoff Direction"], "Bullish")
        self.assertEqual(enriched.iloc[0]["Alignment"], "Agree")
        self.assertEqual(enriched.iloc[1]["Alignment"], "Conflict")


if __name__ == "__main__":
    unittest.main()
