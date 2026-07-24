"""Archived pandas/hmmlearn wrappers retained behind compatibility APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "MacroHMMTrainingConfig": ("macro_training", "MacroHMMTrainingConfig"),
    "MacroHMMTrainingResult": ("macro_training", "MacroHMMTrainingResult"),
    "MarketGMMHMM": ("hmmlearn_models", "MarketGMMHMM"),
    "MarketHMM": ("hmmlearn_models", "MarketHMM"),
    "build_macro_hmm_emissions": (
        "macro_training",
        "build_macro_hmm_emissions",
    ),
    "train_macro_hmm": ("macro_training", "train_macro_hmm"),
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
