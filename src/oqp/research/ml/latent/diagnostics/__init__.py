"""Lazy public API for latent-representation diagnostics."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "codebook_health_summary": ("codebook", "codebook_health_summary"),
    "compute_code_target_ic": ("codebook", "compute_code_target_ic"),
    "compute_code_transition_stats": ("codebook", "compute_code_transition_stats"),
    "compute_codebook_usage": ("codebook", "compute_codebook_usage"),
    "compute_gmm_overlap": ("codebook", "compute_gmm_overlap"),
    "compute_manual_feature_profile": ("codebook", "compute_manual_feature_profile"),
    "merge_gmm_probabilities": ("codebook", "merge_gmm_probabilities"),
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
