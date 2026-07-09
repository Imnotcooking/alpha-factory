from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_APP = REPO_ROOT / "apps" / "research_dashboard"
if str(RESEARCH_APP) not in sys.path:
    sys.path.insert(0, str(RESEARCH_APP))

from config import ALPHA_RUNTIME_ARTIFACT_ROOT, ALPHA_RUNTIME_DATA_ROOT, BASE_DIR, DB_PATH  # noqa: E402
from data_manager import DataManager  # noqa: E402
from views.factor_promotion_view import FactorPromotionView  # noqa: E402
from views.system_health_view import SystemHealthView  # noqa: E402
from oqp.research.latent import (  # noqa: E402
    compute_gmm_overlap,
    load_saved_latents,
    merge_gmm_probabilities,
)
from oqp.research.ml import (  # noqa: E402
    FeatureGovernanceConfig,
    compute_feature_governance,
    list_matrix_files,
    load_matrix,
)
from oqp.research.state_space import (  # noqa: E402
    OpportunityScanConfig,
    list_daily_price_files,
    normalize_daily_market_frame,
    run_opportunity_scan,
)
from oqp.data.runtime_paths import default_futures_cn_index_daily_file  # noqa: E402
from oqp.risk.factor_breadth import RiskBreadthConfig, compute_risk_factor_breadth  # noqa: E402


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        raise unittest.SkipTest(f"local research runtime artifact is missing: {path}")


class ResearchDashboardRealDataSmokeTests(unittest.TestCase):
    def test_factor_promotion_and_strategy_data_load(self) -> None:
        board, detail = FactorPromotionView()._load_board()
        runs = DataManager().get_all_runs()

        self.assertGreater(len(board), 0)
        self.assertEqual(len(board), len(detail))
        self.assertGreater(len(runs), 0)
        self.assertIn("run_id", runs.columns)

    def test_adaptive_relationship_lab_runtime_data_smoke(self) -> None:
        files = list_daily_price_files(BASE_DIR)
        if not files:
            raise unittest.SkipTest("no local daily price files available for relationship lab")

        market = normalize_daily_market_frame(pd.read_parquet(files[0]))
        scan = run_opportunity_scan(
            market,
            OpportunityScanConfig(
                min_observations=60,
                lookback=120,
                zscore_window=30,
                max_assets=8,
                min_abs_correlation=0.0,
            ),
        )

        self.assertGreater(len(market), 0)
        self.assertIn("candidates", scan)
        self.assertIn("metadata", scan)

    def test_feature_governance_lab_runtime_data_smoke(self) -> None:
        files = list_matrix_files(BASE_DIR)
        if not files:
            raise unittest.SkipTest("no local feature matrices available for governance lab")

        matrix = load_matrix(files[0])
        target_candidates = [col for col in matrix.columns if col.startswith("target_")]
        if not target_candidates:
            target_candidates = [col for col in matrix.columns if "target" in col.lower()]
        if not target_candidates:
            raise unittest.SkipTest("local feature matrix has no target column")

        result = compute_feature_governance(
            matrix,
            FeatureGovernanceConfig(
                target_col=target_candidates[0],
                min_assets_per_day=3,
            ),
        )

        self.assertGreater(len(matrix), 0)
        self.assertIn("summary", result)
        self.assertGreater(len(result["summary"]), 0)

    def test_regime_lab_runtime_artifacts_smoke(self) -> None:
        matrix_path = Path(ALPHA_RUNTIME_DATA_ROOT) / "feature_store" / "ML_Feature_Matrix.parquet"
        regimes_path = Path(ALPHA_RUNTIME_DATA_ROOT) / "regime" / "GMM_Rolling_Probabilities.parquet"
        latent_dir = Path(ALPHA_RUNTIME_ARTIFACT_ROOT) / "latent_factors"
        _skip_if_missing(matrix_path)
        _skip_if_missing(regimes_path)
        _skip_if_missing(latent_dir)

        feature_preview = pd.read_parquet(matrix_path, columns=["date", "ticker", "close"]).head(1000)
        regimes = pd.read_parquet(regimes_path)
        latent_result = load_saved_latents(latent_dir)
        latent = latent_result.get("latent", pd.DataFrame())

        self.assertGreater(len(feature_preview), 0)
        self.assertGreater(len(regimes), 0)
        self.assertGreater(len(latent), 0)

        merged = merge_gmm_probabilities(latent, regimes)
        overlap, _ = compute_gmm_overlap(merged)
        self.assertGreater(len(overlap), 0)

    def test_risk_breadth_and_health_runtime_data_smoke(self) -> None:
        source = default_futures_cn_index_daily_file()
        _skip_if_missing(source)

        breadth = compute_risk_factor_breadth(
            source,
            RiskBreadthConfig(
                min_observations=120,
                rolling_window=252,
                rolling_step=126,
            ),
        )
        snapshot = SystemHealthView._load_snapshot(str(BASE_DIR), str(DB_PATH))

        self.assertGreater(breadth["metrics"]["valid_assets"], 0)
        self.assertGreater(len(breadth["spectrum"]), 0)
        self.assertIn("checks", snapshot)
        self.assertIn("markets", snapshot)
        self.assertIn("api_readiness", snapshot)
        self.assertIn("data_folders", snapshot)
        api_readiness = snapshot["api_readiness"]
        self.assertIn("provider", api_readiness.columns)
        self.assertIn("asset_class", api_readiness.columns)
        self.assertTrue(api_readiness["provider"].isin(["FMP", "Massive", "Yahoo Finance"]).any())
        folders = snapshot["data_folders"]
        self.assertIn("asset_class", folders.columns)
        self.assertIn("timeframe", folders.columns)
        self.assertIn("latest_update", folders.columns)
        self.assertTrue(
            ((folders["asset_class"] == "FUTURES_CN") & (folders["timeframe"] == "daily")).any()
        )


if __name__ == "__main__":
    unittest.main()
