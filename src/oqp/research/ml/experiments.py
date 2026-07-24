"""Compatibility imports for the supervised-regression experiment ledger.

New code should import from :mod:`oqp.research.ml.regression.experiments`.
"""

from oqp.research.ml.regression.experiments import (
    ML_EXPERIMENT_TABLE,
    MLExperimentResult,
    ensure_ml_experiment_table,
    latest_ml_experiment,
    list_ml_experiments,
    mean_daily_rank_ic,
    new_experiment_id,
    persist_ml_experiment,
    register_failed_ml_experiment,
    register_ml_experiment,
)

__all__ = [
    "ML_EXPERIMENT_TABLE",
    "MLExperimentResult",
    "ensure_ml_experiment_table",
    "latest_ml_experiment",
    "list_ml_experiments",
    "mean_daily_rank_ic",
    "new_experiment_id",
    "persist_ml_experiment",
    "register_failed_ml_experiment",
    "register_ml_experiment",
]
