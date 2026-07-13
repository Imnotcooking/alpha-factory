from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"
RUNTIME_DATA = REPO_ROOT / "runtime" / "data"
FUTURES_CN_TICK_DATA = REPO_ROOT / "runtime" / "data" / "futures_cn" / "tick"
RUNTIME_DB = REPO_ROOT / "runtime" / "db" / "research" / "research_memory.db"
EXPECTED_PAGE_ORDER = [
    "01_Data_Health.py",
    "02_Pattern_Lab.py",
    "03_Intraday_Event_Study.py",
    "04_Arbitrage_Lab.py",
    "05_Regime_Analysis.py",
    "06_Market_Breadth_Lab.py",
    "07_Feature_Review.py",
    "08_Factor_Review.py",
    "09_Strategy_Comparison.py",
]


if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))


def _element_values(elements) -> list[str]:
    return [str(getattr(element, "value", element)) for element in elements]


class ResearchDashboardPageAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required_paths = [
            RUNTIME_DB,
            RUNTIME_DATA / "feature_store" / "ML_Feature_Matrix.parquet",
            RUNTIME_DATA / "regime" / "GMM_Rolling_Probabilities.parquet",
        ]
        missing = [str(path.relative_to(REPO_ROOT)) for path in required_paths if not path.exists()]
        if missing:
            raise unittest.SkipTest(f"local migrated research runtime data is missing: {', '.join(missing)}")
        if not list(FUTURES_CN_TICK_DATA.glob("*tick*.parquet")):
            raise unittest.SkipTest("local migrated tick parquet data is missing")

        try:
            from streamlit.testing.v1 import AppTest  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"Streamlit AppTest unavailable: {exc}") from exc

    def test_research_pages_render_without_errors(self) -> None:
        from streamlit.testing.v1 import AppTest

        page_paths = sorted((RESEARCH_APP / "pages").glob("*.py"))
        self.assertEqual(EXPECTED_PAGE_ORDER, [path.name for path in page_paths])

        for page_path in [RESEARCH_APP / "Homepage.py", *page_paths]:
            with self.subTest(page=page_path.name):
                app = AppTest.from_file(str(page_path))
                app.run(timeout=90)
                failures = _element_values(app.exception) + _element_values(app.error)
                self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
