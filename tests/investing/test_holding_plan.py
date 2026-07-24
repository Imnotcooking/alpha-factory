from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.portfolio import add_holding_plan_columns, load_holding_styles, save_holding_styles


class HoldingPlanTests(unittest.TestCase):
    def test_styles_round_trip_and_atr_levels_follow_position_direction(self) -> None:
        dates = pd.date_range("2026-01-01", periods=20, freq="D")
        history = pd.DataFrame(
            {
                "symbol": ["TEST"] * 20,
                "date": dates,
                "high": range(102, 122),
                "low": range(98, 118),
                "close": range(100, 120),
            }
        )
        holdings = pd.DataFrame(
            [
                {"Broker": "ibkr", "Symbol": "TEST", "Asset Class": "equity", "Quantity": 2, "Average Cost": 80, "Market Price": 100},
                {"Broker": "ibkr", "Symbol": "TEST", "Asset Class": "equity", "Quantity": -2, "Average Cost": 120, "Market Price": 100},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            save_holding_styles({"ibkr|TEST": "Trading"}, path)
            styles = load_holding_styles(path)

        result = add_holding_plan_columns(holdings, history, styles=styles)

        self.assertEqual(result["Holding Style"].tolist(), ["Trading", "Trading"])
        self.assertGreater(float(result.iloc[0]["TP 1x ATR"]), 100)
        self.assertLess(float(result.iloc[0]["SL 1x ATR"]), 100)
        self.assertLess(float(result.iloc[1]["TP 1x ATR"]), 100)
        self.assertGreater(float(result.iloc[1]["SL 1x ATR"]), 100)

    def test_options_do_not_receive_equity_atr_levels(self) -> None:
        holdings = pd.DataFrame(
            [{"Broker": "manual", "Symbol": "TEST", "Asset Class": "option", "Quantity": 1, "Average Cost": 5, "Market Price": 10}]
        )
        history = pd.DataFrame(
            {"symbol": ["TEST"] * 20, "high": [11] * 20, "low": [9] * 20, "close": [10] * 20}
        )
        result = add_holding_plan_columns(holdings, history)
        self.assertEqual(result.iloc[0]["Holding Style"], "Trading")
        self.assertTrue(pd.isna(result.iloc[0]["TP 1x ATR"]))
        self.assertTrue(pd.isna(result.iloc[0]["SL 1x ATR"]))


if __name__ == "__main__":
    unittest.main()
