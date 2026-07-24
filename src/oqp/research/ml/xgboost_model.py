"""Compatibility imports for the XGBoost regression trainer.

New code should import from :mod:`oqp.research.ml.tree_based.xgboost`.
"""

from oqp.research.ml.tree_based.xgboost import (
    XGBoostModelConfig,
    XGBoostRegressorTrainer,
    XGBoostTrainingEngine,
)

__all__ = [
    "XGBoostModelConfig",
    "XGBoostRegressorTrainer",
    "XGBoostTrainingEngine",
]
