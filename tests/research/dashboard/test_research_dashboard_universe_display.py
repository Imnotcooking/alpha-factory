from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"
if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from universe_display import summarize_traded_universe, traded_universe_detail


class ResearchDashboardUniverseDisplayTests(unittest.TestCase):
    def test_full_stored_universe_displays_all(self) -> None:
        tickers = "SSE.600000,SSE.600004,SZSE.000001"

        self.assertEqual(summarize_traded_universe(tickers, 3), "ALL")
        self.assertEqual(
            traded_universe_detail(tickers, 3),
            "ALL (3 assets)",
        )

    def test_selected_subset_remains_visible(self) -> None:
        tickers = "SSE.600000,SZSE.000001"

        self.assertEqual(
            traded_universe_detail(tickers, 5533),
            "SSE.600000, SZSE.000001 (Total Pool: 5,533)",
        )

    def test_long_selected_subset_is_counted_and_truncated(self) -> None:
        tickers = ",".join(f"SSE.{600000 + idx}" for idx in range(20))

        label = summarize_traded_universe(tickers, 5533, max_tickers=3)

        self.assertTrue(label.startswith("20 selected: SSE.600000, SSE.600001, SSE.600002"))
        self.assertTrue(label.endswith(", ..."))


if __name__ == "__main__":
    unittest.main()
