from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    ManualPortfolioInputs,
    value_portfolio_snapshot,
)


class PortfolioValuationTests(unittest.TestCase):
    def test_values_positions_manual_assets_and_cash(self) -> None:
        positions = pd.DataFrame(
            [
                {
                    "date": "2026-06-24",
                    "broker": "Futubull",
                    "ticker": "AAPL",
                    "asset_type": "Equity",
                    "shares": 10,
                    "avg_cost": 100,
                    "current_price": 115,
                    "unrealized_pnl": 0,
                    "currency": "USD",
                    "delta": 1,
                    "gamma": 0,
                },
                {
                    "date": "2026-06-24",
                    "broker": "IBKR Live",
                    "ticker": "MSFT",
                    "asset_type": "Equity",
                    "shares": 2,
                    "avg_cost": 50,
                    "current_price": 55,
                    "unrealized_pnl": 10,
                    "currency": "USD",
                    "delta": 1,
                    "gamma": 0,
                },
            ]
        )
        market_history = pd.DataFrame(
            {
                "AAPL": [100, 110, 120],
                "QQQ": [100, 105, 110],
                "EURUSD=X": [1.2, 1.2, 1.2],
                "GBPUSD=X": [1.3, 1.3, 1.3],
                "CNYUSD=X": [0.14, 0.14, 0.14],
                "HKDUSD=X": [0.128, 0.128, 0.128],
                "GC=F": [1900, 1950, 2000],
            },
            index=pd.to_datetime(["2026-06-22", "2026-06-23", "2026-06-24"]),
        )

        result = value_portfolio_snapshot(
            positions,
            market_history,
            benchmark="QQQ",
            manual_inputs=ManualPortfolioInputs(
                t212_cash_eur=100,
                cny_mutual_fund=1000,
                cny_mutual_fund_pnl=100,
                cny_gold_grams=1,
                cny_gold_cost=400,
            ),
            asset_preferences={"AAPL": {"Category": "Core Compounding"}},
        )

        aapl = result.asset_summary[result.asset_summary["Ticker"] == "AAPL"].iloc[0]
        self.assertEqual(aapl["Category"], "Core Compounding")
        self.assertEqual(aapl["Current_USD"], 1200)
        self.assertEqual(aapl["PnL_USD"], 200)
        self.assertAlmostEqual(result.total_cash, 120.0)
        self.assertAlmostEqual(result.total_net_worth, result.asset_summary["Current_USD"].sum())
        self.assertIn("Beta_to_QQQ", result.asset_summary.columns)

        broker_rows = {
            row["Broker"]: row["Current_USD"]
            for row in result.broker_summary.to_dict("records")
        }
        self.assertEqual(broker_rows["Futubull"], 1200)
        self.assertEqual(broker_rows["IBKR Live"], 110)
        self.assertEqual(broker_rows["Trading212"], 120)

    def test_empty_snapshot_returns_empty_contract_frames(self) -> None:
        result = value_portfolio_snapshot(
            pd.DataFrame(),
            pd.DataFrame(),
            benchmark="SPY",
        )

        self.assertEqual(result.total_net_worth, 0.0)
        self.assertEqual(result.portfolio_beta, 0.0)
        self.assertIn("Beta_to_SPY", result.asset_summary.columns)


if __name__ == "__main__":
    unittest.main()
