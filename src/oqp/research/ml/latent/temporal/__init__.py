"""Lazy public API for reusable temporal latent-representation transforms."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "TemporalPanelConfig": ("panel", "TemporalPanelConfig"),
    "build_cross_sectional_snapshot": ("panel", "build_cross_sectional_snapshot"),
    "build_temporal_windows": ("panel", "build_temporal_windows"),
    "detect_storm_feature_columns": ("panel", "detect_storm_feature_columns"),
    "flatten_temporal_windows": ("panel", "flatten_temporal_windows"),
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
    return sorted((*globals(), *__all__))
