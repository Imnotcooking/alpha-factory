"""Canonical umbrella for reusable machine-learning research components.

The package root is deliberately lazy.  Importing a focused implementation
such as :mod:`oqp.research.ml.regimes.gaussian_hmm` must not initialize the
supervised data stack or an unrelated optional framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Generic feature governance.
    "FeatureGovernanceConfig": (
        "features.governance",
        "FeatureGovernanceConfig",
    ),
    "coerce_numeric_columns": (
        "features.governance",
        "coerce_numeric_columns",
    ),
    "compute_feature_governance": (
        "features.governance",
        "compute_feature_governance",
    ),
    "detect_feature_columns": (
        "features.governance",
        "detect_feature_columns",
    ),
    "infer_feature_matrix_asset_class": (
        "features.governance",
        "infer_feature_matrix_asset_class",
    ),
    "list_matrix_files": ("features.governance", "list_matrix_files"),
    "load_matrix": ("features.governance", "load_matrix"),
    "observed_feature_matrix_asset_classes": (
        "features.governance",
        "observed_feature_matrix_asset_classes",
    ),
    "prepare_feature_matrix_taxonomy": (
        "features.governance",
        "prepare_feature_matrix_taxonomy",
    ),
    "scope_feature_matrix": ("features.governance", "scope_feature_matrix"),
    "tag_feature_family": ("features.governance", "tag_feature_family"),
    # Supervised regression contract and experiment ledger.  The historical
    # flat modules remain compatibility paths during the package migration.
    "BaseMLModel": ("regression.base", "BaseMLModel"),
    "SupervisedModelBase": ("regression.base", "SupervisedModelBase"),
    "ValidationConfig": ("regression.base", "ValidationConfig"),
    "WalkForwardConfig": ("regression.base", "WalkForwardConfig"),
    "MLExperimentResult": ("regression.experiments", "MLExperimentResult"),
    "ensure_ml_experiment_table": (
        "regression.experiments",
        "ensure_ml_experiment_table",
    ),
    "latest_ml_experiment": ("regression.experiments", "latest_ml_experiment"),
    "list_ml_experiments": ("regression.experiments", "list_ml_experiments"),
    "mean_daily_rank_ic": ("regression.experiments", "mean_daily_rank_ic"),
    "persist_ml_experiment": (
        "regression.experiments",
        "persist_ml_experiment",
    ),
    "register_failed_ml_experiment": (
        "regression.experiments",
        "register_failed_ml_experiment",
    ),
    "register_ml_experiment": (
        "regression.experiments",
        "register_ml_experiment",
    ),
    # Tree-based supervised estimators and runtime helpers.
    "LGBMModel": ("tree_based.lightgbm", "LGBMModel"),
    "LGBMModelConfig": ("tree_based.lightgbm", "LGBMModelConfig"),
    "LightGBMRegressorTrainer": (
        "tree_based.lightgbm",
        "LightGBMRegressorTrainer",
    ),
    "XGBoostModelConfig": ("tree_based.xgboost", "XGBoostModelConfig"),
    "XGBoostTrainingEngine": (
        "tree_based.xgboost",
        "XGBoostTrainingEngine",
    ),
    "XGBoostRegressorTrainer": (
        "tree_based.xgboost",
        "XGBoostRegressorTrainer",
    ),
    "MLModelFactory": ("tree_based.factory", "MLModelFactory"),
    "resolve_model_artifact_path": (
        "tree_based.inference",
        "resolve_model_artifact_path",
    ),
    "ModelRuntimeStatus": ("tree_based.runtime", "ModelRuntimeStatus"),
    "probe_model_runtime": ("tree_based.runtime", "probe_model_runtime"),
    "require_model_runtime": ("tree_based.runtime", "require_model_runtime"),
    # Evaluation and optimizer utilities.
    "PurgedMDAConfig": ("evaluation.oos_mda", "PurgedMDAConfig"),
    "build_purged_time_folds": (
        "evaluation.oos_mda",
        "build_purged_time_folds",
    ),
    "compute_oos_mda": ("evaluation.oos_mda", "compute_oos_mda"),
    "default_xgb_regressor": (
        "evaluation.oos_mda",
        "default_xgb_regressor",
    ),
    "rank_ic_score": ("evaluation.oos_mda", "rank_ic_score"),
    "SUPPORTED_TORCH_OPTIMIZERS": (
        "core.optimizers",
        "SUPPORTED_TORCH_OPTIMIZERS",
    ),
    "TrainingOptimizerSpec": ("core.optimizers", "TrainingOptimizerSpec"),
    "build_torch_optimizer": ("core.optimizers", "build_torch_optimizer"),
    # Exact public identities for unsupervised sequential models.
    "GaussianHMM": ("regimes.gaussian_hmm", "GaussianHMM"),
    "GMMHMM": ("regimes.gmm_hmm", "GMMHMM"),
    "StudentTHMM": ("regimes.student_t_hmm", "StudentTHMM"),
    # Online adaptive state-space estimators.
    "DualKalmanRegression": (
        "state_space.dual_kalman_regression",
        "DualKalmanRegression",
    ),
    "DualKalmanRegressionConfig": (
        "state_space.dual_kalman_regression",
        "DualKalmanRegressionConfig",
    ),
    # Dependency-light inventory (implemented models, not fitted artifacts).
    "MODEL_CATALOG_VERSION": ("catalog", "MODEL_CATALOG_VERSION"),
    "research_experiment_catalog": ("catalog", "research_experiment_catalog"),
    "research_model_catalog": ("catalog", "research_model_catalog"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public objects only when callers request them."""

    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - Python's normal module protocol
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy names to interactive tools and documentation builders."""

    return sorted((*globals(), *_LAZY_EXPORTS))
