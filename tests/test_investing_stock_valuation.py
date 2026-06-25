from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.investing import (
    calculate_cagr,
    calculate_dcf_valuation,
    load_stock_watchlist,
    normalize_watchlist,
    save_stock_watchlist,
)


class InvestingStockValuationTests(unittest.TestCase):
    def test_watchlist_normalization_and_runtime_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "watchlist.json"
            save_stock_watchlist([" aapl ", "MSFT", "aapl", "", None], path)
            loaded = load_stock_watchlist(path, legacy_path=None)

        self.assertEqual(loaded, ["AAPL", "MSFT"])

    def test_normalize_watchlist_preserves_order(self) -> None:
        self.assertEqual(normalize_watchlist(["spy", "QQQ", "SPY"]), ["SPY", "QQQ"])

    def test_calculate_cagr_handles_bad_inputs(self) -> None:
        self.assertEqual(calculate_cagr(pd.Series([0, 10, 20])), 0.0)
        self.assertGreater(calculate_cagr(pd.Series([100, 121])), 0.20)

    def test_standard_dcf_valuation_bridge(self) -> None:
        result = calculate_dcf_valuation(
            {
                "fcf_ttm": 100.0,
                "total_cash": 20.0,
                "total_debt": 10.0,
                "shares": 10.0,
                "price": 12.0,
            },
            model="standard",
            wacc=0.10,
            terminal_growth=0.025,
            fcf_growth_1=0.05,
            fcf_growth_2=0.03,
        )

        self.assertEqual(len(result.future_fcf), 10)
        self.assertGreater(result.fair_value_per_share, 0.0)
        self.assertEqual(result.bridge_rows[-1]["Valuation Metric"], "Equity Value")

    def test_dcf_requires_wacc_above_terminal_growth(self) -> None:
        with self.assertRaises(ValueError):
            calculate_dcf_valuation(
                {"fcf_ttm": 100.0, "shares": 10.0},
                model="standard",
                wacc=0.02,
                terminal_growth=0.025,
            )


if __name__ == "__main__":
    unittest.main()
