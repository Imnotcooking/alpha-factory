from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.market import (
    ensure_market_cache_schema,
    fetch_fmp_history,
    load_cached_market_history,
    load_cached_price_history,
    market_cache_status,
    refresh_yahoo_market_cache,
    refresh_fmp_market_cache,
    write_market_history,
)


class MarketCacheTests(unittest.TestCase):
    def sample_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Adj Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1200, 1300],
            },
            index=pd.date_range("2026-06-26", periods=3, freq="D"),
        )

    def test_writes_and_loads_yfinance_shaped_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market.sqlite3"
            rows = write_market_history(" aapl ", self.sample_history(), path=db_path)
            loaded = load_cached_market_history("AAPL", path=db_path)

        self.assertEqual(rows, 3)
        self.assertEqual(list(loaded.columns), ["Open", "High", "Low", "Close", "Adj Close", "Volume"])
        self.assertEqual(float(loaded.iloc[-1]["Close"]), 102.5)

    def test_loads_long_form_cached_price_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market.sqlite3"
            write_market_history("AAPL", self.sample_history(), path=db_path)
            loaded = load_cached_price_history(["AAPL", "MSFT"], path=db_path)

        self.assertEqual(list(loaded.columns), ["symbol", "date", "close"])
        self.assertEqual(loaded["symbol"].unique().tolist(), ["AAPL"])
        self.assertEqual(float(loaded.iloc[-1]["close"]), 102.5)

    def test_status_reports_missing_and_fresh_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market.sqlite3"
            ensure_market_cache_schema(db_path)
            write_market_history("AAPL", self.sample_history(), path=db_path)
            status = market_cache_status(["AAPL", "MSFT"], path=db_path)

        states = dict(zip(status["Symbol"], status["State"]))
        self.assertEqual(states["AAPL"], "fresh")
        self.assertEqual(states["MSFT"], "missing")

    def test_refresh_uses_injected_provider(self) -> None:
        calls: list[tuple[str, str]] = []

        def provider(symbol: str, period: str) -> pd.DataFrame:
            calls.append((symbol, period))
            return self.sample_history()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market.sqlite3"
            result = refresh_yahoo_market_cache(["AAPL"], path=db_path, period="1y", provider=provider)
            loaded = load_cached_market_history("AAPL", path=db_path)

        self.assertEqual(calls, [("AAPL", "1y")])
        self.assertEqual(result.iloc[0]["Status"], "ok")
        self.assertEqual(len(loaded), 3)

    def test_fetch_fmp_history_uses_stable_eod_payload(self) -> None:
        calls: list[tuple[str, bool, dict[str, object]]] = []

        class FakeAdapter:
            def __init__(self, key: str) -> None:
                self.key = key

            def get_json(self, endpoint: str, *, stable: bool, params: dict[str, object]):
                calls.append((endpoint, stable, params))
                return [
                    {"date": "2026-07-10", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 500}
                ]

        history = fetch_fmp_history("spy", "2y", "secret", adapter_factory=FakeAdapter)

        self.assertEqual(len(history), 1)
        self.assertEqual(calls[0][0], "historical-price-eod/full")
        self.assertTrue(calls[0][1])
        self.assertEqual(calls[0][2]["symbol"], "SPY")
        self.assertIn("from", calls[0][2])

    def test_refresh_fmp_cache_writes_fmp_source(self) -> None:
        def provider(symbol: str, period: str, key: str) -> pd.DataFrame:
            self.assertEqual((symbol, period, key), ("SPY", "2y", "secret"))
            return self.sample_history()

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "market.sqlite3"
            result = refresh_fmp_market_cache(
                ["SPY"], api_key="secret", path=db_path, provider=provider
            )
            loaded = load_cached_market_history("SPY", path=db_path, source="fmp")

        self.assertEqual(result.iloc[0]["Status"], "ok")
        self.assertEqual(len(loaded), 3)


if __name__ == "__main__":
    unittest.main()
