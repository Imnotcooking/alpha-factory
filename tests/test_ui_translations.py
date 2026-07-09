from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from oqp.ui import (
    RESEARCH_PAGE_TEXT,
    RESEARCH_TEXT,
    normalize_language,
    ops_tabs,
    ops_text,
    research_page_legacy_catalog,
    research_page_tabs,
    research_page_text,
    research_tabs,
    research_text,
    tr,
)
from oqp.ui.research_dashboard_config import TEXT, get_plotly_template


REPO_ROOT = Path(__file__).resolve().parents[1]


class UiTranslationTests(unittest.TestCase):
    def test_language_aliases_and_inline_translation(self) -> None:
        self.assertEqual(normalize_language("ZH"), "zh")
        self.assertEqual(normalize_language("CN"), "zh")
        self.assertEqual(normalize_language("English"), "en")
        self.assertEqual(tr("ZH", "Save", "保存"), "保存")

    def test_ops_catalog_translates_buttons_and_tabs(self) -> None:
        self.assertEqual(ops_text("zh", "save_daily_note"), "保存每日笔记")
        self.assertEqual(ops_text("en", "load_valuation"), "Load Valuation")
        self.assertEqual(ops_tabs("zh", "journal_tabs")[0], "每日日志")
        self.assertEqual(len(ops_tabs("zh", "option_scanner_tabs")), 9)
        self.assertEqual(len(ops_tabs("zh", "live_tabs")), 5)
        self.assertEqual(len(ops_tabs("zh", "paper_tabs")), 7)
        self.assertEqual(len(ops_tabs("zh", "risk_tabs")), 6)
        self.assertEqual(len(ops_tabs("zh", "execution_tabs")), 5)
        self.assertEqual(len(ops_tabs("zh", "workbench_tabs")), 6)
        self.assertEqual(len(ops_tabs("zh", "workbench_option_tabs")), 5)

    def test_research_catalog_uses_shared_translation_module(self) -> None:
        self.assertIn("title", TEXT["EN"])
        self.assertIn("title", TEXT["ZH"])
        self.assertEqual(TEXT["ZH"]["title"], RESEARCH_TEXT["zh"]["title"])
        self.assertEqual(research_text("ZH", "title"), TEXT["ZH"]["title"])
        self.assertEqual(research_tabs("zh", "metrics")[0], TEXT["ZH"]["metrics"][0])
        self.assertEqual(get_plotly_template("DARK"), "plotly_dark")

    def test_specialized_research_page_catalogs_are_shared(self) -> None:
        expected_pages = {
            "adaptive_relationship_lab",
            "tick_pulse_lab",
            "pulse_discovery_lab",
            "regime_characterisation_lab",
            "alpha_feature_governance",
            "risk_factor_breadth_lab",
        }

        self.assertTrue(expected_pages.issubset(RESEARCH_PAGE_TEXT))
        for page_key in expected_pages:
            catalog = RESEARCH_PAGE_TEXT[page_key]
            self.assertEqual(set(catalog), {"en", "zh"})
            self.assertEqual(set(catalog["en"]), set(catalog["zh"]))

        self.assertEqual(
            research_page_text("tick_pulse_lab", "ZH", "title"),
            "Tick 事件研究",
        )
        self.assertEqual(
            research_page_tabs("adaptive_relationship_lab", "zh", "tabs")[0],
            "雷达",
        )

    def test_legacy_research_page_text_wrappers_still_match_shared_catalog(
        self,
    ) -> None:
        adaptive_legacy = research_page_legacy_catalog("adaptive_relationship_lab")
        tick_legacy = research_page_legacy_catalog("tick_pulse_lab")

        self.assertEqual(
            adaptive_legacy["ZH"]["title"],
            RESEARCH_PAGE_TEXT["adaptive_relationship_lab"]["zh"]["title"],
        )
        self.assertIn("criteria_feature_names", tick_legacy["EN"])

        wrapper_paths = {
            "adaptive_relationship_lab": REPO_ROOT
            / "apps"
            / "research_dashboard"
            / "arbitrage_lab"
            / "text.py",
            "tick_pulse_lab": REPO_ROOT
            / "apps"
            / "research_dashboard"
            / "tick_pulse_lab"
            / "text.py",
        }
        for page_key, path in wrapper_paths.items():
            spec = importlib.util.spec_from_file_location(
                f"_{page_key}_text_test", path
            )
            self.assertIsNotNone(spec)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(
                module.PAGE_TEXT["EN"]["title"],
                RESEARCH_PAGE_TEXT[page_key]["en"]["title"],
            )

    def test_research_dashboard_config_keeps_local_runtime_paths(self) -> None:
        config_path = REPO_ROOT / "apps" / "research_dashboard" / "config.py"
        spec = importlib.util.spec_from_file_location(
            "_alpha_ui_config_test", config_path
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.TEXT["ZH"]["title"], TEXT["ZH"]["title"])
        self.assertTrue(
            module.DB_PATH.endswith("runtime/db/research/research_memory.db")
        )
        self.assertTrue(
            module.LOGS_DIR.endswith("runtime/artifacts/research")
        )


if __name__ == "__main__":
    unittest.main()
