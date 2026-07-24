"""Compatibility import for tree-based model artifact resolution.

New code should import from :mod:`oqp.research.ml.tree_based.inference`.
"""

from oqp.research.ml.tree_based.inference import resolve_model_artifact_path

__all__ = ["resolve_model_artifact_path"]
