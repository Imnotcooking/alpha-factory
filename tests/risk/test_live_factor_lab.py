from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oqp.risk import (
    LiveFactorLabConfig,
    combine_price_histories,
    compute_factor_proxy_lab,
    compute_pca_crowding_lab,
    factor_proxy_symbols,
)


class LiveFactorLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2025-01-01", periods=120, freq="B")
        self.price_history = self._price_history(
            {
                "AAPL": 0.0010,
                "MSFT": 0.0008,
                "SPY": 0.0007,
                "IWM": 0.0009,
                "IWD": 0.0006,
                "IWF": 0.0008,
                "MTUM": 0.0011,
                "QUAL": 0.00075,
            }
        )
        self.exposure = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Economic Exposure": [6000.0, 4000.0],
            }
        )
        self.config = LiveFactorLabConfig(lookback_days=1000, min_observations=40)

    def test_factor_proxy_lab_runs_from_cached_prices(self) -> None:
        result = compute_factor_proxy_lab(self.exposure, self.price_history, config=self.config)

        self.assertEqual(result["status"], "live")
        self.assertFalse(result["betas"].empty)
        self.assertIn("Market (SPY)", result["betas"]["Factor"].tolist())
        self.assertGreaterEqual(float(result["summary"].loc[result["summary"]["Metric"].eq("R squared"), "Value"].iloc[0]), 0.0)

    def test_pca_lab_reports_component_spectrum(self) -> None:
        result = compute_pca_crowding_lab(self.exposure, self.price_history, config=self.config)

        self.assertEqual(result["status"], "live")
        self.assertFalse(result["spectrum"].empty)
        self.assertFalse(result["top_drivers"].empty)
        self.assertIn("PC1", result["spectrum"]["Component"].tolist())

    def test_combines_and_canonicalizes_price_history_sources(self) -> None:
        first = self.price_history[self.price_history["symbol"].isin(["AAPL"])]
        second = pd.DataFrame(
            {
                "ticker": ["tcehy"],
                "date": [self.dates[0]],
                "price": [50.0],
            }
        )

        combined = combine_price_histories(first, second)

        self.assertIn("AAPL", combined["symbol"].tolist())
        self.assertIn("TENCENT", combined["symbol"].tolist())

    def test_proxy_symbol_list_is_unique(self) -> None:
        symbols = factor_proxy_symbols()
        self.assertEqual(len(symbols), len(set(symbols)))

    def _price_history(self, drifts: dict[str, float]) -> pd.DataFrame:
        rows = []
        for symbol, drift in drifts.items():
            seasonal = np.sin(np.arange(len(self.dates)) / 9.0) * 0.002
            returns = drift + seasonal
            close = 100.0 * np.cumprod(1.0 + returns)
            for date, value in zip(self.dates, close, strict=False):
                rows.append({"symbol": symbol, "date": date, "close": float(value)})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
