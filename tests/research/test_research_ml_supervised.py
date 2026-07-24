from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from oqp.research import MLModelFactory as PublicMLModelFactory
from oqp.research.artifacts import ModelArtifactStore
from oqp.research.ml import (
    LGBMModel,
    LGBMModelConfig,
    MLExperimentResult,
    MLModelFactory,
    SupervisedModelBase,
    ValidationConfig,
    WalkForwardConfig,
    XGBoostModelConfig,
    XGBoostTrainingEngine,
    list_ml_experiments,
)


class DummySupervisedModel(SupervisedModelBase):
    def train(self) -> tuple:
        return ()


class _FakeLGBDataset:
    def __init__(self, data, label, reference=None, **kwargs):
        self.data = data
        self.label = label
        self.reference = reference


class _FakeLGBModel:
    def __init__(self, feature_count: int):
        self.feature_count = feature_count

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        return np.arange(1, self.feature_count + 1, dtype=float)

    def predict(self, frame) -> np.ndarray:
        return np.asarray(frame, dtype=float)[:, 0]


def _fake_lightgbm_module():
    return types.SimpleNamespace(
        Dataset=_FakeLGBDataset,
        train=lambda params, train_set, **kwargs: _FakeLGBModel(
            train_set.data.shape[1]
        ),
        early_stopping=lambda **kwargs: ("early_stopping", kwargs),
        log_evaluation=lambda **kwargs: ("log_evaluation", kwargs),
    )


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
        self.assertEqual(MLModelFactory.supported_models(), ("lightgbm", "xgboost"))
        self.assertEqual(MLModelFactory.normalize_model_type("LGBM"), "lightgbm")
        self.assertEqual(MLModelFactory.normalize_model_type("XGB"), "xgboost")
        with self.assertRaises(ValueError):
            MLModelFactory.create_model("UNKNOWN", "matrix.parquet")

    def test_fixed_date_validation_uses_shared_purge_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "features.parquet"
            self._feature_matrix(days=12, assets=3).to_parquet(matrix_path)
            model = DummySupervisedModel(
                matrix_path,
                validation_config=ValidationConfig(
                    mode="fixed_date",
                    split_date="2026-01-12",
                    purge_gap_days=2,
                ),
            )
            train_df, test_df, _ = next(model.generate_validation_folds())

        self.assertLess(train_df["date"].max(), pd.Timestamp("2026-01-10"))
        self.assertGreaterEqual(test_df["date"].min(), pd.Timestamp("2026-01-12"))
        self.assertEqual(model.validation_policy()["mode"], "fixed_date")

    def test_model_adapters_return_one_experiment_contract(self) -> None:
        validation = ValidationConfig(
            mode="walk_forward",
            min_train_days=8,
            test_window_days=3,
            purge_gap_days=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = Path(tmp) / "features.parquet"
            self._feature_matrix(days=16, assets=4).to_parquet(matrix_path)
            lgbm = LGBMModel(
                matrix_path,
                config=LGBMModelConfig(
                    validation=validation,
                    params={
                        "objective": "regression",
                        "metric": "rmse",
                        "verbosity": -1,
                        "seed": 42,
                    },
                    num_boost_round=8,
                    early_stopping_rounds=2,
                ),
            )
            xgb = XGBoostTrainingEngine(
                matrix_path,
                config=XGBoostModelConfig(
                    target_col="target_1d_rank",
                    validation=validation,
                    params={
                        "n_estimators": 8,
                        "max_depth": 2,
                        "learning_rate": 0.1,
                        "objective": "reg:squarederror",
                        "random_state": 42,
                        "n_jobs": 1,
                    },
                ),
                factor_id="fac_test",
            )

            with patch.dict("sys.modules", {"lightgbm": _fake_lightgbm_module()}):
                lgbm_result = lgbm.train()
            results = [lgbm_result, xgb.train()]

        for result in results:
            self.assertIsInstance(result, MLExperimentResult)
            self.assertEqual(
                list(result.predictions.columns),
                ["date", "ticker", "target", "prediction", "fold"],
            )
            self.assertEqual(
                list(result.feature_importance.columns),
                ["feature", "importance"],
            )
            self.assertEqual(result.validation_policy["mode"], "walk_forward")
            self.assertGreater(result.metrics["fold_count"], 0)

    def test_xgboost_run_persists_model_registry_and_experiment_ledger(self) -> None:
        validation = ValidationConfig(
            mode="walk_forward",
            min_train_days=8,
            test_window_days=4,
            purge_gap_days=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_path = root / "features.parquet"
            model_path = root / "models" / "model.json"
            importance_path = root / "importance" / "importance.csv"
            predictions_path = root / "predictions" / "predictions.parquet"
            db_path = root / "research.db"
            self._feature_matrix(days=14, assets=4).to_parquet(matrix_path)
            trainer = XGBoostTrainingEngine(
                matrix_path,
                config=XGBoostModelConfig(
                    target_col="target_1d_rank",
                    validation=validation,
                    params={
                        "n_estimators": 5,
                        "max_depth": 2,
                        "objective": "reg:squarederror",
                        "random_state": 42,
                        "n_jobs": 1,
                    },
                ),
                model_name="fac_test_xgboost",
                factor_id="fac_test",
                asset_class="FUTURES_CN",
                model_output_path=model_path,
                importance_output_path=importance_path,
                predictions_output_path=predictions_path,
                registry_db_path=db_path,
                artifact_store=ModelArtifactStore(
                    root_dir=root / "registry_artifacts",
                    workspace_root=root,
                ),
            )

            result = trainer.run()
            rows = list_ml_experiments(db_path, factor_id="fac_test")

            self.assertTrue(model_path.exists())
            self.assertTrue(importance_path.exists())
            self.assertTrue(predictions_path.exists())
            self.assertEqual(result.status, "completed")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_id"], result.experiment_id)
            self.assertEqual(rows[0]["asset_class"], "FUTURES_CN")
            self.assertTrue(rows[0]["predictions_path"].endswith("predictions.parquet"))

    def test_failed_training_is_visible_in_experiment_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix_path = root / "features.parquet"
            db_path = root / "research.db"
            self._feature_matrix(days=6, assets=3).drop(
                columns=["target_1d_rank"]
            ).to_parquet(matrix_path)
            trainer = XGBoostTrainingEngine(
                matrix_path,
                target_col="target_1d_rank",
                factor_id="fac_failed",
                registry_db_path=db_path,
            )

            with self.assertRaisesRegex(ValueError, "missing target"):
                trainer.run()
            rows = list_ml_experiments(db_path, factor_id="fac_failed")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("target_1d_rank", rows[0]["error"])

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
                        "asset_class": "FUTURES_CN",
                        "f_signal": signal,
                        "prob_regime": signal / 10,
                        "target_1d_rank": signal + 0.01,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
