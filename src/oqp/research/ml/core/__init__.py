"""Framework-neutral training infrastructure shared by ML model families."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "SUPPORTED_TORCH_OPTIMIZERS": ("optimizers", "SUPPORTED_TORCH_OPTIMIZERS"),
    "TrainingOptimizerSpec": ("optimizers", "TrainingOptimizerSpec"),
    "build_torch_optimizer": ("optimizers", "build_torch_optimizer"),
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
