from __future__ import annotations

import json
import unittest
from datetime import date

import pandas as pd

from oqp.options import (
    extract_portfolio_option_legs,
    option_greeks_frame,
    option_payoff_curve,
    option_risk_summary,
)


class OptionPayoffTests(unittest.TestCase):
    def test_vertical_spread_payoff_uses_signed_leg_costs(self) -> None:
        row = {
            "symbol": "AVGO 410/430C",
            "asset_class": "option_spread",
            "quantity": 1,
            "average_cost": 8.95,
            "market_price": 8.95,
            "market_value": 895.0,
            "unrealized_pnl": 0.0,
            "multiplier": 100,
            "underlying": "AVGO",
            "expiry": "2027-01-15",
            "option_type": "call",
            "metadata_json": json.dumps(
                {
                    "legs": [
                        {"side": "buy", "option_type": "call", "strike": 410, "quantity": 1, "average_cost": 78.37},
                        {"side": "sell", "option_type": "call", "strike": 430, "quantity": -1, "average_cost": 69.42},
                    ]
                }
            ),
        }

        curve = option_payoff_curve(row, points=3, lower=410, upper=430)
        risk = option_risk_summary(row, today=date(2026, 7, 2))
        by_metric = dict(zip(risk["Metric"], risk["Value"], strict=False))

        self.assertEqual(len(extract_portfolio_option_legs(row)), 2)
        self.assertAlmostEqual(float(curve.iloc[0]["Expiry P&L"]), -895.0)
        self.assertAlmostEqual(float(curve.iloc[-1]["Expiry P&L"]), 1105.0)
        self.assertAlmostEqual(float(by_metric["Entry Debit"]), 895.0)
        self.assertAlmostEqual(float(by_metric["Max Loss"]), -895.0)
        self.assertAlmostEqual(float(by_metric["Max Profit"]), 1105.0)
        self.assertEqual(by_metric["DTE"], 197)

    def test_single_long_call_summary(self) -> None:
        row = {
            "symbol": "JD271217C00035000",
            "asset_class": "option",
            "quantity": 2,
            "average_cost": 5.85,
            "market_price": 2.82,
            "market_value": 563.0,
            "unrealized_pnl": -607.0,
            "multiplier": 100,
            "underlying": "JD",
            "expiry": "2027-12-17",
            "option_type": "call",
            "strike": 35,
            "metadata_json": json.dumps({"delta": 0.4, "implied_volatility": 0.55}),
        }

        risk = option_risk_summary(pd.Series(row), today=date(2026, 7, 2))
        greeks = option_greeks_frame(row)
        by_metric = dict(zip(risk["Metric"], risk["Value"], strict=False))

        self.assertAlmostEqual(float(by_metric["Entry Debit"]), 1170.0)
        self.assertAlmostEqual(float(by_metric["Current P&L"]), -606.0)
        self.assertEqual(greeks.iloc[0]["Delta"], 0.4)
        self.assertEqual(greeks.iloc[0]["IV"], 0.55)


if __name__ == "__main__":
    unittest.main()
