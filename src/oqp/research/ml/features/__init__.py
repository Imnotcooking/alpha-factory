"""Feature discovery and governance utilities for reusable ML workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    name: ("governance", name)
    for name in (
        "DEFAULT_CORR_MIN_PERIODS",
        "DEFAULT_CORR_THRESHOLD",
        "DEFAULT_MAX_CORR_ROWS",
        "DEFAULT_MIN_ASSETS_PER_DAY",
        "EXPLICIT_ENGINEERED_FEATURES",
        "FAMILY_RULES",
        "FEATURE_PREFIXES",
        "FeatureGovernanceConfig",
        "LEGACY_MATRIX_ASSET_CLASS",
        "LEGACY_MATRIX_NAMES",
        "UNKNOWN_ASSET_CLASS",
        "coerce_numeric_columns",
        "compute_feature_governance",
        "detect_feature_columns",
        "infer_feature_matrix_asset_class",
        "list_matrix_files",
        "load_matrix",
        "observed_feature_matrix_asset_classes",
        "prepare_feature_matrix_taxonomy",
        "scope_feature_matrix",
        "tag_feature_family",
    )
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
