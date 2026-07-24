"""Compatibility imports for the LightGBM regression trainer.

New code should import from :mod:`oqp.research.ml.tree_based.lightgbm`.
"""

from oqp.research.ml.tree_based.lightgbm import (
    LGBMModel,
    LGBMModelConfig,
    LightGBMRegressorTrainer,
)

__all__ = [
    "LGBMModel",
    "LGBMModelConfig",
    "LightGBMRegressorTrainer",
]
