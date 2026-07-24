"""Canonical VQ-VAE numerical core with dependency-lazy public imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "FittedVQVAE": ("model", "FittedVQVAE"),
    "VQEncoding": ("model", "VQEncoding"),
    "VQTrainingEpoch": ("model", "VQTrainingEpoch"),
    "VQVAE_ARTIFACT_VERSION": ("serialization", "VQVAE_ARTIFACT_VERSION"),
    "VQVAE_CORE_VERSION": ("model", "VQVAE_CORE_VERSION"),
    "VQVAEConfig": ("config", "VQVAEConfig"),
    "VQVAEError": ("model", "VQVAEError"),
    "VQVAETrainer": ("model", "VQVAETrainer"),
    "load_vqvae_bundle": ("serialization", "load_vqvae_bundle"),
    "save_vqvae_bundle": ("serialization", "save_vqvae_bundle"),
    "torch_available": ("network", "torch_available"),
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
