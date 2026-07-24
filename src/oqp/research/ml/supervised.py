"""Compatibility imports for the supervised regression task contract.

New code should import from :mod:`oqp.research.ml.regression.base`.
"""

from oqp.research.ml.regression.base import (
    BaseMLModel,
    SupervisedModelBase,
    ValidationConfig,
    WalkForwardConfig,
)

__all__ = [
    "BaseMLModel",
    "SupervisedModelBase",
    "ValidationConfig",
    "WalkForwardConfig",
]
