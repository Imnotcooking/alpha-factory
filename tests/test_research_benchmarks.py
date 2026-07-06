from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from oqp.research import BenchmarkFactory as PublicBenchmarkFactory
from oqp.research.backtesting import (
    BENCHMARK_RETURN_COL,
    AbsoluteReturnBenchmark,
    BenchmarkFactory,
    BuyAndHoldBenchmark,
    CSI300Benchmark,
    RiskFreeRateBenchmark,
    SPYBenchmark,
    SectorNeutralBenchmark,
    TickerBenchmark,
    dynamic_equal_weight_benchmark,
)


class ResearchBenchmarkTests(unittest.TestCase):
    def test_absolute_return_benchmark_normalizes_unique_dates(self) -> None:
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01 14:30:00",
                        "2026-01-01 09:00:00",
                        "2026-01-02 00:00:00",
                    ]
                ),
                "ticker": ["A", "B", "A"],
            }
        )

        benchmark = AbsoluteReturnBenchmark(ann_rate=0.05).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-01", "2026-01-02"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            np.repeat((1.05 ** (1.0 / 252.0)) - 1.0, 2),
        )

    def test_buy_and_hold_uses_only_active_traded_tickers_and_dates(self) -> None:
        raw = self._raw_prices()
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-02"]),
                "ticker": ["A", "A", "B"],
                "target_weight": [1.0, 0.5, 0.0],
            }
        )

        benchmark = BuyAndHoldBenchmark(raw).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02", "2026-01-03"],
        )
        np.testing.assert_allclose(benchmark[BENCHMARK_RETURN_COL].to_numpy(), [0.10, 0.10])

    def test_equal_weight_benchmark_can_average_active_assets(self) -> None:
        raw = self._raw_prices()
        active_matrix = pd.DataFrame(
            {
                "A": [True, True],
                "B": [True, False],
            },
            index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
        )

        benchmark = dynamic_equal_weight_benchmark(raw, active_matrix=active_matrix)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02", "2026-01-03"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            [0.0, 0.10],
            atol=1e-12,
        )

    def test_buy_and_hold_normalizes_raw_ticker_ids_before_active_filter(self) -> None:
        raw = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"]
                ),
                "ticker": [101, 101, 202, 202],
                "close": [100.0, 110.0, 200.0, 180.0],
            }
        )
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02"]),
                "ticker": ["101"],
                "target_weight": [1.0],
            }
        )

        benchmark = BuyAndHoldBenchmark(raw).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02"],
        )
        np.testing.assert_allclose(benchmark[BENCHMARK_RETURN_COL].to_numpy(), [0.10])

    def test_buy_and_hold_falls_back_to_full_universe_without_active_weights(self) -> None:
        raw = self._raw_prices()
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "ticker": ["A", "B"],
                "target_weight": [0.0, 0.0],
            }
        )

        benchmark = BuyAndHoldBenchmark(raw).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02", "2026-01-03", "2026-01-04"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            [0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_buy_and_hold_falls_back_to_absolute_when_close_is_missing(self) -> None:
        raw = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "ticker": ["A", "A"],
                "open": [100.0, 101.0],
            }
        )
        strategy = raw[["date", "ticker"]]

        benchmark = BuyAndHoldBenchmark(raw).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-01", "2026-01-02"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            np.repeat((1.05 ** (1.0 / 252.0)) - 1.0, 2),
        )

    def test_factory_preserves_alpha_lab_benchmark_type_strings(self) -> None:
        raw = self._raw_prices()

        self.assertIs(PublicBenchmarkFactory, BenchmarkFactory)
        self.assertIsInstance(
            BenchmarkFactory.create_benchmark("ABSOLUTE"),
            AbsoluteReturnBenchmark,
        )
        self.assertIsInstance(
            BenchmarkFactory.create_benchmark("RISK_FREE"),
            RiskFreeRateBenchmark,
        )
        self.assertIsInstance(
            BenchmarkFactory.create_benchmark("BUY_AND_HOLD", raw_df=raw),
            BuyAndHoldBenchmark,
        )
        self.assertIsInstance(BenchmarkFactory.create_benchmark("UNKNOWN"), AbsoluteReturnBenchmark)
        with self.assertRaises(ValueError):
            BenchmarkFactory.create_benchmark("BUY_AND_HOLD")

    def test_ticker_benchmark_tracks_named_external_index_on_strategy_dates(self) -> None:
        raw = self._index_prices()
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                "ticker": ["A", "A"],
                "target_weight": [1.0, 1.0],
            }
        )

        benchmark = TickerBenchmark(raw, "SPY").generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02", "2026-01-03"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            [0.02, -0.005],
        )

    def test_named_index_factory_aliases_return_semantic_benchmark_classes(self) -> None:
        raw = self._index_prices()

        spy = BenchmarkFactory.create_benchmark("SP500", raw_df=raw)
        csi300 = BenchmarkFactory.create_benchmark("CSI_300", raw_df=raw)
        custom = BenchmarkFactory.create_benchmark(
            "INDEX",
            raw_df=raw,
            benchmark_tickers=["SPY", "000300.SH"],
        )

        self.assertIsInstance(spy, SPYBenchmark)
        self.assertIsInstance(csi300, CSI300Benchmark)
        self.assertIsInstance(custom, TickerBenchmark)

    def test_ticker_benchmark_raises_when_required_index_data_is_missing(self) -> None:
        strategy = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"])})

        with self.assertRaises(ValueError):
            TickerBenchmark(self._raw_prices(), "SPY").generate(strategy)

    def test_sector_neutral_benchmark_equal_weights_active_sectors(self) -> None:
        raw = self._sector_prices()
        strategy = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-02",
                        "2026-01-02",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-03",
                        "2026-01-03",
                    ]
                ),
                "ticker": ["A", "B", "C", "A", "B", "C"],
                "target_weight": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            }
        )

        benchmark = SectorNeutralBenchmark(raw).generate(strategy)

        self.assertEqual(
            list(benchmark["date"].dt.strftime("%Y-%m-%d")),
            ["2026-01-02", "2026-01-03"],
        )
        np.testing.assert_allclose(
            benchmark[BENCHMARK_RETURN_COL].to_numpy(),
            [0.10, 0.15],
        )

    @staticmethod
    def _raw_prices() -> pd.DataFrame:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
        return pd.DataFrame(
            {
                "date": list(dates) * 2,
                "ticker": ["A"] * 4 + ["B"] * 4,
                "close": [
                    100.0,
                    110.0,
                    121.0,
                    133.1,
                    200.0,
                    180.0,
                    162.0,
                    145.8,
                ],
            }
        )

    @staticmethod
    def _index_prices() -> pd.DataFrame:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        return pd.DataFrame(
            {
                "date": list(dates) * 2,
                "ticker": ["SPY"] * 3 + ["000300.SH"] * 3,
                "close": [100.0, 102.0, 101.49, 4000.0, 4040.0, 4080.4],
            }
        )

    @staticmethod
    def _sector_prices() -> pd.DataFrame:
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
        return pd.DataFrame(
            {
                "date": list(dates) * 3,
                "ticker": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
                "sector": ["Tech"] * 6 + ["Energy"] * 3,
                "close": [
                    100.0,
                    110.0,
                    121.0,
                    100.0,
                    90.0,
                    81.0,
                    100.0,
                    120.0,
                    144.0,
                ],
            }
        )


if __name__ == "__main__":
    unittest.main()
