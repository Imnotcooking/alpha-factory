"""Boundary tests for the educational supervised-ML package taxonomy."""

from __future__ import annotations

from oqp.research.ml.experiments import MLExperimentResult as LegacyExperimentResult
from oqp.research.ml.inference import (
    resolve_model_artifact_path as legacy_resolve_model_artifact_path,
)
from oqp.research.ml.lgbm_model import LGBMModel as LegacyLGBMModel
from oqp.research.ml.model_factory import MLModelFactory as LegacyMLModelFactory
from oqp.research.ml.regression.base import ValidationConfig
from oqp.research.ml.regression.experiments import MLExperimentResult
from oqp.research.ml.runtime import probe_model_runtime as legacy_probe_model_runtime
from oqp.research.ml.tree_based.factory import MLModelFactory
from oqp.research.ml.tree_based.inference import resolve_model_artifact_path
from oqp.research.ml.tree_based.lightgbm import (
    LGBMModel,
    LightGBMRegressorTrainer,
)
from oqp.research.ml.tree_based.runtime import probe_model_runtime
from oqp.research.ml.tree_based.xgboost import (
    XGBoostRegressorTrainer,
    XGBoostTrainingEngine,
)
from oqp.research.ml.xgboost_model import (
    XGBoostTrainingEngine as LegacyXGBoostTrainingEngine,
)


def test_generic_regression_contract_excludes_regime_probabilities_by_default() -> None:
    assert ValidationConfig().include_prob_features is False


def test_historical_trainer_names_are_identity_aliases() -> None:
    assert LGBMModel is LightGBMRegressorTrainer
    assert LegacyLGBMModel is LightGBMRegressorTrainer
    assert XGBoostTrainingEngine is XGBoostRegressorTrainer
    assert LegacyXGBoostTrainingEngine is XGBoostRegressorTrainer


def test_flat_infrastructure_modules_preserve_object_identity() -> None:
    assert LegacyExperimentResult is MLExperimentResult
    assert LegacyMLModelFactory is MLModelFactory
    assert legacy_resolve_model_artifact_path is resolve_model_artifact_path
    assert legacy_probe_model_runtime is probe_model_runtime


def test_factory_returns_exact_named_trainers_without_loading_native_libraries() -> None:
    lightgbm = MLModelFactory.create_model("lgbm", "features.parquet")
    xgboost = MLModelFactory.create_model("xgb", "features.parquet")

    assert type(lightgbm) is LightGBMRegressorTrainer
    assert type(xgboost) is XGBoostRegressorTrainer
    assert lightgbm.model_type == "lightgbm"
    assert xgboost.model_type == "xgboost"
