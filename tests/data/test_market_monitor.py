from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from oqp.market import (
    MarketMonitorInstrument,
    market_monitor_breadth,
    market_monitor_cache_status,
    market_monitor_correlation,
    market_monitor_movers,
    market_monitor_regime,
    market_monitor_snapshot,
    market_monitor_universe,
    refresh_market_monitor_cache,
    write_market_history,
)


class MarketMonitorTests(unittest.TestCase):
    def instrument(self) -> MarketMonitorInstrument:
        return MarketMonitorInstrument(
            "gold", "Gold", "GCUSD", "Commodities", "commodity", "Metal", yahoo_symbol="GC=F"
        )

    def history(self, periods: int = 90) -> pd.DataFrame:
        index = pd.date_range(end="2026-07-10", periods=periods, freq="D")
        close = pd.Series(range(100, 100 + periods), index=index, dtype=float)
        return pd.DataFrame(
            {
                "Open": close - 1,
                "High": close + 1,
                "Low": close - 2,
                "Close": close,
                "Adj Close": close,
                "Volume": 1000,
            },
            index=index,
        )

    def test_default_universe_covers_major_cross_asset_lanes(self) -> None:
        universe = market_monitor_universe()
        asset_classes = {item.asset_class for item in universe}

        self.assertGreaterEqual(len(universe), 30)
        self.assertTrue({"equity", "rates", "credit", "fx", "commodity", "crypto", "volatility"}.issubset(asset_classes))

    def test_refresh_prefers_fmp_and_does_not_call_yahoo(self) -> None:
        yahoo_calls: list[str] = []

        def fmp_provider(symbol: str, period: str, key: str) -> pd.DataFrame:
            self.assertEqual(symbol, "GCUSD")
            return self.history()

        def yahoo_provider(symbol: str, period: str) -> pd.DataFrame:
            yahoo_calls.append(symbol)
            return self.history()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.sqlite3"
            result = refresh_market_monitor_cache(
                [self.instrument()],
                fmp_api_key="secret",
                path=path,
                fmp_provider=fmp_provider,
                yahoo_provider=yahoo_provider,
            )
            snapshot = market_monitor_snapshot([self.instrument()], path=path)

        self.assertEqual(result.iloc[0]["Provider"], "fmp")
        self.assertEqual(yahoo_calls, [])
        self.assertEqual(snapshot.iloc[0]["Source"], "fmp")

    def test_refresh_falls_back_to_yahoo_under_canonical_symbol(self) -> None:
        yahoo_calls: list[str] = []

        def empty_fmp(symbol: str, period: str, key: str) -> pd.DataFrame:
            return pd.DataFrame()

        def yahoo_provider(symbol: str, period: str) -> pd.DataFrame:
            yahoo_calls.append(symbol)
            return self.history()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.sqlite3"
            result = refresh_market_monitor_cache(
                [self.instrument()],
                fmp_api_key="secret",
                path=path,
                fmp_provider=empty_fmp,
                yahoo_provider=yahoo_provider,
            )
            snapshot = market_monitor_snapshot([self.instrument()], path=path)

        self.assertEqual(yahoo_calls, ["GC=F"])
        self.assertEqual(result.iloc[0]["Status"], "fallback")
        self.assertEqual(snapshot.iloc[0]["Symbol"], "GCUSD")
        self.assertEqual(snapshot.iloc[0]["Source"], "yahoo")
        self.assertGreater(float(snapshot.iloc[0]["20D %"]), 0)

    def test_cache_health_prefers_fresh_fmp_over_yahoo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.sqlite3"
            write_market_history("GCUSD", self.history(), path=path, source="yahoo")
            write_market_history("GCUSD", self.history(), path=path, source="fmp")
            status = market_monitor_cache_status([self.instrument()], path=path)

        self.assertEqual(status.iloc[0]["Provider"], "fmp")
        self.assertEqual(status.iloc[0]["State"], "fresh")

    def test_missing_history_has_actionable_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market.sqlite3"
            snapshot = market_monitor_snapshot([self.instrument()], path=path)

        self.assertEqual(snapshot.iloc[0]["Status"], "missing")
        self.assertEqual(snapshot.iloc[0]["Cache State"], "missing")
        self.assertIn("no cached close history", snapshot.iloc[0]["Notes"])

    def test_decision_analytics_rank_breadth_and_movers(self) -> None:
        snapshot = pd.DataFrame(
            [
                {"Instrument": "S&P", "Symbol": "SPY", "Asset Class": "equity", "20D %": 0.08, "Status": "ok", "Source": "fmp"},
                {"Instrument": "Nasdaq", "Symbol": "QQQ", "Asset Class": "equity", "20D %": -0.02, "Status": "ok", "Source": "fmp"},
                {"Instrument": "Gold", "Symbol": "GCUSD", "Asset Class": "commodity", "20D %": 0.04, "Status": "ok", "Source": "fmp"},
            ]
        )
        breadth = market_monitor_breadth(snapshot, horizon="20D %").set_index("Asset Class")
        winners, losers = market_monitor_movers(snapshot, horizon="20D %", count=1)

        self.assertEqual(float(breadth.at["equity", "Breadth"]), 0.5)
        self.assertEqual(breadth.at["equity", "Best Symbol"], "SPY")
        self.assertEqual(winners.iloc[0]["Symbol"], "SPY")
        self.assertEqual(losers.iloc[0]["Symbol"], "QQQ")

    def test_regime_is_explainable_and_risk_on(self) -> None:
        snapshot = pd.DataFrame(
            [
                {"Symbol": symbol, "20D %": 0.05, "Last": 100.0}
                for symbol in ("SPY", "QQQ", "IWM", "HYG", "EEM", "MCHI", "HGUSD", "BTCUSD")
            ]
            + [{"Symbol": "^VIX", "20D %": -0.10, "Last": 16.0}]
        )
        regime = market_monitor_regime(snapshot)

        self.assertEqual(regime["Label"], "Risk On")
        self.assertEqual(len(regime["Components"]), 4)
        self.assertAlmostEqual(float(regime["Risk Breadth"]), 1.0)

    def test_correlation_aligns_cached_daily_returns(self) -> None:
        index = pd.date_range("2026-01-01", periods=40, freq="D")
        base = pd.Series(range(100, 140), index=index, dtype=float)
        correlation = market_monitor_correlation(
            {"A": pd.DataFrame({"Close": base}), "B": pd.DataFrame({"Close": base * 2})},
            lookback=30,
            min_observations=20,
        )

        self.assertEqual(correlation.shape, (2, 2))
        self.assertAlmostEqual(float(correlation.at["A", "B"]), 1.0)


if __name__ == "__main__":
    unittest.main()
