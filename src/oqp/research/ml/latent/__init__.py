"""Latent-representation models in the reusable research ML library.

The public symbols are resolved lazily so importing the taxonomy does not make
PyTorch a mandatory dependency for unrelated model families.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "FittedVQVAE": ("vqvae", "FittedVQVAE"),
    "VQEncoding": ("vqvae", "VQEncoding"),
    "VQTrainingEpoch": ("vqvae", "VQTrainingEpoch"),
    "VQVAE_ARTIFACT_VERSION": ("vqvae", "VQVAE_ARTIFACT_VERSION"),
    "VQVAE_CORE_VERSION": ("vqvae", "VQVAE_CORE_VERSION"),
    "VQVAEConfig": ("vqvae", "VQVAEConfig"),
    "VQVAEError": ("vqvae", "VQVAEError"),
    "VQVAETrainer": ("vqvae", "VQVAETrainer"),
    "load_vqvae_bundle": ("vqvae", "load_vqvae_bundle"),
    "save_vqvae_bundle": ("vqvae", "save_vqvae_bundle"),
    "torch_available": ("vqvae", "torch_available"),
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
