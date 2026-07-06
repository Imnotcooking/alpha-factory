from __future__ import annotations

import json
import unittest
from datetime import date

import pandas as pd

from oqp.options import option_book_summary, option_position_diagnostics


class OptionBookDiagnosticsTests(unittest.TestCase):
    def test_book_summary_scales_live_greeks_and_vol_premium(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "SPY",
                    "asset_class": "equity",
                    "quantity": 10,
                    "market_price": 500.0,
                    "market_value": 5000.0,
                },
                {
                    "symbol": "SPY270117C00490000",
                    "asset_class": "option",
                    "quantity": 2,
                    "average_cost": 15.0,
                    "market_price": 18.0,
                    "market_value": 3600.0,
                    "unrealized_pnl": 600.0,
                    "multiplier": 100,
                    "metadata_json": json.dumps(
                        {
                            "iv": 0.25,
                            "delta": 0.60,
                            "gamma": 0.02,
                            "theta": -0.05,
                            "vega": 0.20,
                        }
                    ),
                },
            ]
        )
        hv = pd.DataFrame([{"symbol": "SPY", "hv_20d": 0.20}])

        diagnostics = option_position_diagnostics(positions, hv, today=date(2026, 7, 2))
        summary = option_book_summary(positions, hv, today=date(2026, 7, 2))

        self.assertEqual(len(diagnostics), 1)
        self.assertAlmostEqual(float(diagnostics.iloc[0]["Intrinsic"]), 10.0)
        self.assertAlmostEqual(float(diagnostics.iloc[0]["Extrinsic"]), 8.0)
        self.assertAlmostEqual(float(diagnostics.iloc[0]["Delta Units"]), 120.0)
        self.assertAlmostEqual(float(diagnostics.iloc[0]["Delta Dollars"]), 60_000.0)
        self.assertAlmostEqual(float(diagnostics.iloc[0]["Gamma $ 1%"]), 50.0)
        self.assertEqual(diagnostics.iloc[0]["Quality Flag"], "ok")

        row = summary.iloc[0]
        self.assertEqual(row["Option Rows"], 1)
        self.assertEqual(row["Option Legs"], 1)
        self.assertAlmostEqual(float(row["Contracts"]), 2.0)
        self.assertAlmostEqual(float(row["Weighted IV"]), 0.25)
        self.assertAlmostEqual(float(row["IV/HV"]), 1.25)
        self.assertAlmostEqual(float(row["Vega / 1 vol"]), 40.0)

    def test_position_diagnostics_can_solve_iv_and_fallback_greeks(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "symbol": "QQQ",
                    "asset_class": "etf",
                    "quantity": 5,
                    "market_price": 400.0,
                    "market_value": 2000.0,
                },
                {
                    "symbol": "QQQ270117P00390000",
                    "asset_class": "option",
                    "quantity": 1,
                    "average_cost": 9.0,
                    "market_price": 12.0,
                    "market_value": 1200.0,
                    "multiplier": 100,
                    "metadata_json": "{}",
                },
            ]
        )

        diagnostics = option_position_diagnostics(positions, today=date(2026, 7, 2))

        self.assertEqual(len(diagnostics), 1)
        self.assertGreater(float(diagnostics.iloc[0]["IV"]), 0.0)
        self.assertLess(float(diagnostics.iloc[0]["Delta"]), 0.0)
        self.assertEqual(diagnostics.iloc[0]["Model Source"], "bsm_fallback")


if __name__ == "__main__":
    unittest.main()
