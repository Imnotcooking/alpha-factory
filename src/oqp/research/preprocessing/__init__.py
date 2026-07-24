"""Compatibility package for :mod:`oqp.research.ml.preprocessing`.

The historical path remains dependency-lazy and returns the identical
canonical classes and functions.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_CANONICAL_PACKAGE = "oqp.research.ml.preprocessing"
__all__ = [
    "FittedMatrixPreprocessor",
    "MissingValuePolicy",
    "PREPROCESSOR_CORE_VERSION",
    "PreprocessingError",
    "PreprocessingSpec",
    "dump_preprocessor_json",
    "fit_matrix_preprocessor",
    "hash_numeric_matrix",
    "load_preprocessor_json",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_CANONICAL_PACKAGE), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
