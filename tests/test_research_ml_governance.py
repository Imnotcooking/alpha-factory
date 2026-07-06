from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from oqp.research.ml import (
    FeatureGovernanceConfig,
    PurgedMDAConfig,
    build_purged_time_folds,
    compute_feature_governance,
    compute_oos_mda,
    detect_feature_columns,
    tag_feature_family,
)


class ResearchMLGovernanceTests(unittest.TestCase):
    def test_feature_governance_identifies_quality_redundancy_and_keepers(self) -> None:
        panel = self._feature_panel(days=80, assets=6)

        result = compute_feature_governance(
            panel,
            FeatureGovernanceConfig(
                target_col="target_1d_rank",
                corr_threshold=0.95,
                min_assets_per_day=4,
                corr_min_periods=50,
                random_state=11,
            ),
        )

        summary = result["summary"].set_index("feature")
        corr_pairs = result["corr_pairs"]
        clusters = result["clusters"]

        self.assertEqual(result["metadata"]["features"], 4)
        self.assertIn("f_mom", summary.index)
        self.assertGreater(summary.loc["f_mom", "abs_mean_ic"], summary.loc["f_noise", "abs_mean_ic"])
        self.assertFalse(corr_pairs.empty)
        self.assertTrue(
            (
                (corr_pairs["feature_a"] == "f_mom")
                & (corr_pairs["feature_b"] == "f_mom_clone")
            ).any()
            or (
                (corr_pairs["feature_a"] == "f_mom_clone")
                & (corr_pairs["feature_b"] == "f_mom")
            ).any()
        )
        self.assertFalse(clusters[clusters["cluster_size"] > 1].empty)
        self.assertFalse(result["keeper_features"].empty)
        self.assertFalse(result["pca_variance"].empty)
        self.assertEqual(tag_feature_family("f_mom_20d"), "Momentum")

    def test_feature_detection_can_exclude_regime_probability_columns(self) -> None:
        panel = self._feature_panel(days=10, assets=4)

        with_probs = detect_feature_columns(panel, include_prob_features=True)
        without_probs = detect_feature_columns(panel, include_prob_features=False)

        self.assertIn("prob_regime_bull", with_probs)
        self.assertNotIn("prob_regime_bull", without_probs)
        self.assertIn("f_mom", without_probs)

    def test_purged_time_folds_remove_embargo_window_around_test_dates(self) -> None:
        panel = self._feature_panel(days=40, assets=4)
        folds = build_purged_time_folds(
            panel,
            n_splits=4,
            embargo_days=3,
            min_train_rows=20,
            min_test_rows=10,
        )

        self.assertTrue(folds)
        dates = pd.to_datetime(panel["date"])
        for fold in folds:
            train_dates = dates.iloc[fold["train_idx"]]
            forbidden_start = fold["test_start"] - pd.Timedelta(days=3)
            forbidden_end = fold["test_end"] + pd.Timedelta(days=3)
            self.assertFalse(train_dates.between(forbidden_start, forbidden_end).any())

    def test_oos_mda_ranks_real_feature_above_noise(self) -> None:
        panel = self._feature_panel(days=100, assets=6)
        estimator = RandomForestRegressor(
            n_estimators=60,
            max_depth=4,
            random_state=11,
            n_jobs=1,
        )

        result = compute_oos_mda(
            panel,
            feature_cols=["f_mom", "f_noise"],
            target_col="target_1d_rank",
            estimator=estimator,
            config=PurgedMDAConfig(
                n_splits=4,
                embargo_days=3,
                max_features=2,
                max_rows=10_000,
                min_train_rows=80,
                min_test_rows=40,
                min_assets_per_day=3,
                random_state=11,
            ),
        )

        summary = result["summary"].set_index("feature")
        self.assertGreater(summary.loc["f_mom", "mda_mean"], summary.loc["f_noise", "mda_mean"])
        self.assertGreater(summary.loc["f_mom", "mda_mean"], 0)
        self.assertGreaterEqual(result["metadata"]["folds"], 2)
        self.assertIn("gain_detail", result)

    @staticmethod
    def _feature_panel(days: int = 90, assets: int = 6) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2026-01-01", periods=days, freq="B")
        rows = []
        for day_idx, date in enumerate(dates):
            common = np.sin(day_idx / 8.0)
            for asset_idx in range(assets):
                useful = common + asset_idx * 0.3
                noise = rng.normal()
                rows.append(
                    {
                        "date": date,
                        "ticker": f"a{asset_idx}",
                        "f_mom": useful,
                        "f_mom_clone": useful * 1.01 + rng.normal(scale=0.001),
                        "f_noise": noise,
                        "prob_regime_bull": 1.0 / (1.0 + np.exp(-common)),
                        "target_1d_rank": useful + rng.normal(scale=0.02),
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
