from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"

if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from data_manager import DataManager  # noqa: E402
from views.ml_view import MLView  # noqa: E402


class FakeDataManager:
    def __init__(self, has_importance: bool = False):
        self.has_importance = has_importance
        self.calls = []

    def has_feature_importance(
        self,
        run_id: str,
        factor_id: str | None = None,
        *,
        include_factor_level: bool = True,
    ) -> bool:
        self.calls.append((run_id, factor_id, include_factor_level))
        return self.has_importance


class ResearchDashboardMLVisibilityTests(unittest.TestCase):
    def test_heuristic_run_without_importance_hides_ml_tab(self) -> None:
        view = MLView(FakeDataManager(has_importance=False))
        metadata = pd.Series(
            {
                "run_id": "run_heuristic",
                "factor_id": "fac_058_CYC_Cost_Trend",
                "name": "CYC Cost Trend for CN Futures",
                "factor_category": "Trend / Cost Basis",
                "execution_mode": "direct",
                "evaluation_geometry": "time_series",
                "stat_research_family": "fac_058_CYC_Cost_Trend",
            }
        )

        self.assertFalse(view.should_render_tab("run_heuristic", metadata))

    def test_importance_artifact_shows_ml_tab(self) -> None:
        fake_dm = FakeDataManager(has_importance=True)
        view = MLView(fake_dm)
        metadata = pd.Series({"factor_id": "fac_054", "name": "Plain label"})

        self.assertTrue(view.should_render_tab("run_ml", metadata))
        self.assertEqual([("run_ml", "fac_054", True)], fake_dm.calls)

    def test_model_run_metadata_shows_ml_tab_even_before_artifact(self) -> None:
        view = MLView(FakeDataManager(has_importance=False))
        metadata = pd.Series(
            {
                "backtest_engine": "run_ml_backtest",
                "model_type": "xgboost_regressor",
            }
        )

        self.assertTrue(view.should_render_tab("run_ml_without_csv", metadata))

    def test_future_multi_factor_model_metadata_can_opt_into_ml_tab(self) -> None:
        view = MLView(FakeDataManager(has_importance=False))
        metadata = pd.Series(
            {
                "factor_id": "strategy_blend_001",
                "backtest_engine": "multi_factor_ml",
                "strategy_type": "allocator_model",
                "model_type": "stacked_meta_model",
            }
        )

        self.assertTrue(view.should_render_tab("run_multifactor", metadata))

    def test_data_manager_detects_run_and_legacy_factor_importance_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            run_dir = tmp / "feature_importance"
            legacy_dir = tmp / "returns" / "feature_importance"
            run_dir.mkdir(parents=True)
            legacy_dir.mkdir(parents=True)
            (run_dir / "feature_importance_run_123.csv").write_text(
                "feature,importance\nf_a,0.7\n",
                encoding="utf-8",
            )
            (legacy_dir / "feature_importance_fac_999.csv").write_text(
                "feature,importance\nf_b,0.3\n",
                encoding="utf-8",
            )

            with patch("data_manager.LOGS_DIR", str(tmp)):
                dm = DataManager(db_path=str(tmp / "empty.db"))

                self.assertTrue(dm.has_feature_importance("run_123", include_factor_level=False))
                self.assertFalse(dm.has_feature_importance("run_missing", include_factor_level=False))
                self.assertTrue(dm.has_feature_importance("run_missing", factor_id="fac_999"))
                loaded = dm.get_feature_importance("run_123")

        self.assertEqual(["feature", "importance"], list(loaded.columns))
        self.assertEqual("f_a", loaded.iloc[0]["feature"])


if __name__ == "__main__":
    unittest.main()
