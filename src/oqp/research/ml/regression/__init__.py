"""Shared contracts and experiment records for supervised regression research.

This package defines the *task* (labelled regression with chronological
validation).  Concrete algorithms, such as LightGBM and XGBoost, live in
``oqp.research.ml.tree_based``.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "BaseMLModel": ("base", "BaseMLModel"),
    "ML_EXPERIMENT_TABLE": ("experiments", "ML_EXPERIMENT_TABLE"),
    "MLExperimentResult": ("experiments", "MLExperimentResult"),
    "SupervisedModelBase": ("base", "SupervisedModelBase"),
    "ValidationConfig": ("base", "ValidationConfig"),
    "WalkForwardConfig": ("base", "WalkForwardConfig"),
    "ensure_ml_experiment_table": ("experiments", "ensure_ml_experiment_table"),
    "latest_ml_experiment": ("experiments", "latest_ml_experiment"),
    "list_ml_experiments": ("experiments", "list_ml_experiments"),
    "mean_daily_rank_ic": ("experiments", "mean_daily_rank_ic"),
    "new_experiment_id": ("experiments", "new_experiment_id"),
    "persist_ml_experiment": ("experiments", "persist_ml_experiment"),
    "register_failed_ml_experiment": (
        "experiments",
        "register_failed_ml_experiment",
    ),
    "register_ml_experiment": ("experiments", "register_ml_experiment"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
