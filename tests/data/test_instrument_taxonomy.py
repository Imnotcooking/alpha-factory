from __future__ import annotations

import math
import unittest

import pandas as pd

from oqp.contracts import (
    ASSET_TAXONOMY,
    MarketVertical,
    market_vertical_spec,
    normalize_market_vertical,
)
from oqp.data import (
    CORE_DASHBOARD_ASSET_CLASSES,
    LANE_METADATA,
    InstrumentMaster,
    asset_class_label,
    attach_asset_class,
    core_dashboard_asset_classes,
    load_asset_taxonomy,
    taxonomy_frame,
    taxonomy_options,
    taxonomy_row,
)
from oqp.research import infer_dataset_tradability


class InstrumentTaxonomyTests(unittest.TestCase):
    def test_market_vertical_contract_normalizes_aliases_and_exposes_metadata(
        self,
    ) -> None:
        self.assertEqual(normalize_market_vertical("china futures"), "FUTURES_CN")
        self.assertEqual(normalize_market_vertical("US equities"), "EQUITY_US")
        self.assertEqual(
            normalize_market_vertical(MarketVertical.OPTIONS_US), "OPTIONS_US"
        )
        self.assertEqual(normalize_market_vertical("Chinese options"), "OPTIONS_CN")
        self.assertEqual(normalize_market_vertical("A shares"), "EQUITY_CN")
        self.assertEqual(normalize_market_vertical(math.nan), "UNKNOWN")

        futures = market_vertical_spec("CN_FUTURES")

        self.assertIsNotNone(futures)
        assert futures is not None
        self.assertEqual(futures.region, "CN")
        self.assertTrue(futures.price_limit)
        self.assertTrue(ASSET_TAXONOMY["FUTURES_CN"]["vectorizable"])
        self.assertFalse(ASSET_TAXONOMY["OPTIONS_US"]["vectorizable"])
        self.assertFalse(ASSET_TAXONOMY["OPTIONS_CN"]["vectorizable"])
        self.assertEqual(
            ASSET_TAXONOMY["OPTIONS_CN"]["backtest_route"], "event_driven_options"
        )

    def test_instrument_master_handles_cn_futures_and_us_equity_without_symbol_damage(
        self,
    ) -> None:
        futures_master = InstrumentMaster("FUTURES_CN")
        gold = futures_master.get_profile("黄金(au)[指数]")
        kq_gold = futures_master.get_profile("KQ.i@SHFE.au")
        kq_csi500 = futures_master.get_profile("KQ.i@CFFEX.IC")
        rebar = futures_master.get_profile("rb2601")
        fibreboard = futures_master.get_profile("fb1901")
        plywood = futures_master.get_profile("bb1901")
        eggs = futures_master.get_profile("jd2601")
        wire_rod = futures_master.get_profile("wr2601")
        legacy_czce = {
            ticker: futures_master.get_profile(ticker)
            for ticker in ("JR", "LR", "PM", "RI", "RS", "WH", "ZC")
        }

        self.assertEqual(gold.ticker, "au")
        self.assertEqual(gold.exchange, "SHFE")
        self.assertEqual(gold.multiplier, 1000)
        self.assertEqual(gold.tick_size, 0.02)
        self.assertEqual(kq_gold.ticker, "au")
        self.assertEqual(kq_gold.exchange, "SHFE")
        self.assertEqual(kq_gold.multiplier, 1000)
        self.assertEqual(kq_gold.fee_open, 20.0)
        self.assertEqual(kq_csi500.ticker, "IC")
        self.assertEqual(kq_csi500.exchange, "CFFEX")
        self.assertEqual(kq_csi500.multiplier, 200)
        self.assertEqual(rebar.ticker, "rb")
        self.assertEqual(rebar.exchange, "SHFE")
        self.assertEqual(fibreboard.exchange, "DCE")
        self.assertEqual(fibreboard.tick_size, 0.05)
        self.assertEqual(plywood.multiplier, 500)
        self.assertEqual(eggs.multiplier, 5)
        self.assertEqual(wire_rod.exchange, "SHFE")
        self.assertEqual(wire_rod.multiplier, 10)
        self.assertEqual(wire_rod.tick_size, 1.0)
        self.assertEqual(
            {
                ticker: (profile.sector, profile.multiplier, profile.tick_size)
                for ticker, profile in legacy_czce.items()
            },
            {
                "JR": ("软商品", 20, 1.0),
                "LR": ("软商品", 20, 1.0),
                "PM": ("软商品", 50, 1.0),
                "RI": ("软商品", 20, 1.0),
                "RS": ("油脂油料", 10, 1.0),
                "WH": ("软商品", 20, 1.0),
                "ZC": ("能源", 100, 0.2),
            },
        )
        self.assertTrue(
            all(profile.exchange == "CZCE" for profile in legacy_czce.values())
        )

        equity = InstrumentMaster("EQUITY_US").get_profile("3M")

        self.assertEqual(equity.ticker, "3M")
        self.assertEqual(equity.exchange, "US")
        self.assertEqual(equity.tick_size, 0.01)

        cn_equity = InstrumentMaster("EQUITY_CN").get_profile("600519")
        us_option = InstrumentMaster("OPTIONS_US").get_profile("AAPL_20270115_C200")
        cn_option = InstrumentMaster("OPTIONS_CN").get_profile("510050C2609M03000")

        self.assertEqual(cn_equity.ticker, "600519")
        self.assertEqual(cn_equity.exchange, "CN")
        self.assertEqual(cn_equity.multiplier, 1)
        self.assertEqual(us_option.exchange, "US")
        self.assertEqual(us_option.multiplier, 100)
        self.assertEqual(cn_option.exchange, "CN")
        self.assertEqual(cn_option.multiplier, 10_000)

    def test_asset_taxonomy_helpers_attach_and_describe_market_verticals(self) -> None:
        taxonomy = load_asset_taxonomy()
        frame = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "au2608", "rb2601"],
                "market_vertical": ["US equities", "EQUITY_US", "CN_FUTURES", None],
            }
        )

        attached = attach_asset_class(frame, default="FUTURES_CN")
        options = taxonomy_options(taxonomy, observed=set(attached["asset_class"]))
        row = taxonomy_row("futures_cn", taxonomy, local_rows=10, local_assets=2)

        self.assertEqual(
            attached["asset_class"].tolist(),
            ["EQUITY_US", "EQUITY_US", "FUTURES_CN", "FUTURES_CN"],
        )
        self.assertEqual(options[0], "FUTURES_CN")
        self.assertIn("EQUITY_US", options)
        self.assertIn("Chinese Futures", asset_class_label("FUTURES_CN", taxonomy))
        self.assertEqual(row["region"], "CN")
        self.assertTrue(row["has_local_regime_data"])

    def test_research_dataset_policy_uses_market_vertical_aliases(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01"]),
                "ticker": ["黄金(au)[指数]"],
                "close": [790.0],
            }
        )

        profile = infer_dataset_tradability(
            frame,
            source_path="Macro_Index_V2.parquet",
            asset_class="china futures",
            data_frequency="daily",
        )

        self.assertEqual(profile.dataset_role, "research_index")
        self.assertEqual(profile.tradability, "research_proxy")

    def test_promoted_taxonomy_objects_are_available_without_lab_wrappers(self) -> None:
        taxonomy = load_asset_taxonomy()

        self.assertEqual(ASSET_TAXONOMY["FUTURES_CN"]["region"], "CN")
        self.assertEqual(taxonomy["OPTIONS_US"]["vectorizable"], False)
        self.assertEqual(taxonomy["OPTIONS_CN"]["instrument_family"], "option")
        self.assertEqual(
            InstrumentMaster("FUTURES_CN").get_profile("au2608").exchange, "SHFE"
        )

    def test_dashboard_taxonomy_core_lanes_include_qmt_china_routes(self) -> None:
        lanes = core_dashboard_asset_classes()

        self.assertEqual(
            lanes,
            ["EQUITY_US", "OPTIONS_US", "EQUITY_CN", "OPTIONS_CN", "FUTURES_CN"],
        )
        self.assertEqual(tuple(lanes), CORE_DASHBOARD_ASSET_CLASSES)
        for asset_class in lanes:
            self.assertIn(asset_class, ASSET_TAXONOMY)

        frame = taxonomy_frame(asset_classes=lanes)
        self.assertEqual(frame["asset_class"].tolist(), lanes)
        self.assertIn(
            "Massive",
            frame.loc[frame["asset_class"].eq("OPTIONS_US"), "provider"].iat[0],
        )
        for asset_class in ["EQUITY_CN", "OPTIONS_CN", "FUTURES_CN"]:
            row = frame.loc[frame["asset_class"].eq(asset_class)].iloc[0]
            self.assertIn("华源证券", row["broker"])
            self.assertIn("QMT", row["execution"])
            self.assertEqual(row["region"], "CN")

        self.assertEqual(LANE_METADATA["EQUITY_CN"]["status"], "planned_qmt")
        self.assertEqual(LANE_METADATA["OPTIONS_CN"]["status"], "planned_qmt")


if __name__ == "__main__":
    unittest.main()
