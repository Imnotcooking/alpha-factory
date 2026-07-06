from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from oqp.research import MLModelFactory as PublicMLModelFactory
from oqp.research.ml import (
    LGBMModel,
    MLModelFactory,
    SupervisedModelBase,
    WalkForwardConfig,
    XGBoostTrainingEngine,
)


class DummySupervisedModel(SupervisedModelBase):
    def train(self) -> tuple:
        return ()


class ResearchMLSupervisedTests(unittest.TestCase):
    def test_supervised_base_detects_features_and_generates_purged_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "features.parquet"
            self._feature_matrix(days=12, assets=3).to_parquet(matrix_path)

            model = DummySupervisedModel(
                matrix_path,
                walk_forward_config=WalkForwardConfig(
                    min_train_days=5,
                    test_window_days=3,
                    purge_gap_days=1,
                    include_prob_features=False,
                ),
            )
            frame, features = model.prepare_supervised_matrix()
            folds = list(model.generate_walk_forward_folds())

        self.assertEqual(features, ["f_signal"])
        self.assertEqual(len(frame), 36)
        self.assertEqual(len(folds), 3)
        train_df, test_df, fold_features = folds[0]
        self.assertEqual(fold_features, ["f_signal"])
        self.assertLess(train_df["date"].max(), test_df["date"].min())
        self.assertEqual(train_df["date"].nunique(), 4)
        self.assertEqual(test_df["date"].nunique(), 3)

    def test_model_factory_routes_without_loading_optional_training_libraries(self) -> None:
        lgbm = MLModelFactory.create_model("LIGHTGBM", "matrix.parquet")
        xgb = MLModelFactory.create_model("XGB", "matrix.parquet")

        self.assertIs(PublicMLModelFactory, MLModelFactory)
        self.assertIsInstance(lgbm, LGBMModel)
        self.assertIsInstance(xgb, XGBoostTrainingEngine)
        self.assertIn("LIGHTGBM", MLModelFactory.supported_models())
        self.assertIn("XGBOOST", MLModelFactory.supported_models())
        with self.assertRaises(ValueError):
            MLModelFactory.create_model("UNKNOWN", "matrix.parquet")

    def test_xgboost_engine_sample_weights_apply_uniqueness_and_time_decay(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4, freq="B")
        train_df = pd.DataFrame({"date": np.repeat(dates, 2)})
        engine = XGBoostTrainingEngine(target_horizon_days=4)

        weights = engine._compute_sample_weights(train_df)

        self.assertEqual(len(weights), len(train_df))
        self.assertGreater(float(weights.iloc[-1]), float(weights.iloc[0]))
        self.assertAlmostEqual(float(weights.iloc[0]), 0.125)
        self.assertLessEqual(float(weights.max()), 0.25)

    @staticmethod
    def _feature_matrix(days: int, assets: int) -> pd.DataFrame:
        rows = []
        dates = pd.date_range("2026-01-01", periods=days, freq="B")
        for day_idx, date in enumerate(dates):
            for asset_idx in range(assets):
                signal = float(day_idx + asset_idx / 10)
                rows.append(
                    {
                        "date": date,
                        "ticker": f"a{asset_idx}",
                        "f_signal": signal,
                        "prob_regime": signal / 10,
                        "target_1d_rank": signal + 0.01,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
