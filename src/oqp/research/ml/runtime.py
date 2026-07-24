"""Compatibility imports for tree-based native-runtime probes.

New code should import from :mod:`oqp.research.ml.tree_based.runtime`.
"""

from oqp.research.ml.tree_based.runtime import (
    ModelRuntimeStatus,
    probe_model_runtime,
    require_model_runtime,
)

__all__ = ["ModelRuntimeStatus", "probe_model_runtime", "require_model_runtime"]
