"""Portable, leakage-safe preprocessing for ordered ML feature matrices."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "FittedMatrixPreprocessor": ("artifact", "FittedMatrixPreprocessor"),
    "MissingValuePolicy": ("artifact", "MissingValuePolicy"),
    "PREPROCESSOR_CORE_VERSION": ("artifact", "PREPROCESSOR_CORE_VERSION"),
    "PreprocessingError": ("artifact", "PreprocessingError"),
    "PreprocessingSpec": ("artifact", "PreprocessingSpec"),
    "fit_matrix_preprocessor": ("artifact", "fit_matrix_preprocessor"),
    "hash_numeric_matrix": ("artifact", "hash_numeric_matrix"),
    "dump_preprocessor_json": ("serialization", "dump_preprocessor_json"),
    "load_preprocessor_json": ("serialization", "load_preprocessor_json"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_LAZY_EXPORTS))
