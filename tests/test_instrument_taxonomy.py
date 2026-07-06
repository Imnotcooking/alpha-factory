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
    InstrumentMaster,
    asset_class_label,
    attach_asset_class,
    load_asset_taxonomy,
    taxonomy_options,
    taxonomy_row,
)
from oqp.research import infer_dataset_tradability


class InstrumentTaxonomyTests(unittest.TestCase):
    def test_market_vertical_contract_normalizes_aliases_and_exposes_metadata(self) -> None:
        self.assertEqual(normalize_market_vertical("china futures"), "FUTURES_CN")
        self.assertEqual(normalize_market_vertical("US equities"), "EQUITY_US")
        self.assertEqual(normalize_market_vertical(MarketVertical.OPTIONS_US), "OPTIONS_US")
        self.assertEqual(normalize_market_vertical(math.nan), "UNKNOWN")

        futures = market_vertical_spec("CN_FUTURES")

        self.assertIsNotNone(futures)
        assert futures is not None
        self.assertEqual(futures.region, "CN")
        self.assertTrue(futures.price_limit)
        self.assertTrue(ASSET_TAXONOMY["FUTURES_CN"]["vectorizable"])
        self.assertFalse(ASSET_TAXONOMY["OPTIONS_US"]["vectorizable"])

    def test_instrument_master_handles_cn_futures_and_us_equity_without_symbol_damage(self) -> None:
        futures_master = InstrumentMaster("FUTURES_CN")
        gold = futures_master.get_profile("黄金(au)[指数]")
        rebar = futures_master.get_profile("rb2601")

        self.assertEqual(gold.ticker, "au")
        self.assertEqual(gold.exchange, "SHFE")
        self.assertEqual(gold.multiplier, 1000)
        self.assertEqual(gold.tick_size, 0.02)
        self.assertEqual(rebar.ticker, "rb")
        self.assertEqual(rebar.exchange, "SHFE")

        equity = InstrumentMaster("EQUITY_US").get_profile("3M")

        self.assertEqual(equity.ticker, "3M")
        self.assertEqual(equity.exchange, "US")
        self.assertEqual(equity.tick_size, 0.01)

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

        self.assertEqual(attached["asset_class"].tolist(), ["EQUITY_US", "EQUITY_US", "FUTURES_CN", "FUTURES_CN"])
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
        self.assertEqual(InstrumentMaster("FUTURES_CN").get_profile("au2608").exchange, "SHFE")


if __name__ == "__main__":
    unittest.main()
