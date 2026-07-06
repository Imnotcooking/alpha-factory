"""Loader utilities for the OQP C++ quant core."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable


PACKAGED_QUANT_CORE = "oqp.native._quant_core"
LEGACY_QUANT_CORE = "quant_core"


class QuantCoreUnavailable(ImportError):
    """Raised when no compatible native quant core can be imported."""


@dataclass(frozen=True, slots=True)
class NativeModuleStatus:
    module_name: str
    available: bool
    source_path: str | None = None
    missing_features: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.available and not self.missing_features


def load_quant_core(
    required_features: Iterable[str] | None = None,
    *,
    allow_legacy: bool = True,
    legacy_paths: Iterable[str | Path] = (),
) -> ModuleType:
    """Import the packaged native kernel, falling back to legacy lab builds.

    New code should build and import ``oqp.native._quant_core``. During the
    alpha-lab migration, callers may pass the old ``cpp_engine`` directory as a
    legacy path so existing local builds still work.
    """

    required = tuple(required_features or ())
    attempts: list[NativeModuleStatus] = []

    for module_name in _candidate_modules(allow_legacy=allow_legacy):
        if module_name == LEGACY_QUANT_CORE:
            _prepend_existing_paths(legacy_paths)

        status = _load_status(module_name, required)
        attempts.append(status)
        if status.ok:
            return importlib.import_module(module_name)

    details = "; ".join(_format_status(status) for status in attempts)
    raise QuantCoreUnavailable(f"No compatible quant core is available: {details}")


def quant_core_status(
    required_features: Iterable[str] | None = None,
    *,
    allow_legacy: bool = True,
    legacy_paths: Iterable[str | Path] = (),
) -> NativeModuleStatus:
    """Return import status without raising."""

    required = tuple(required_features or ())
    last_status: NativeModuleStatus | None = None
    for module_name in _candidate_modules(allow_legacy=allow_legacy):
        if module_name == LEGACY_QUANT_CORE:
            _prepend_existing_paths(legacy_paths)
        status = _load_status(module_name, required)
        if status.ok:
            return status
        last_status = status
    return last_status or NativeModuleStatus(
        module_name=PACKAGED_QUANT_CORE,
        available=False,
        error="no module candidates checked",
    )


def _candidate_modules(*, allow_legacy: bool) -> tuple[str, ...]:
    if allow_legacy:
        return (PACKAGED_QUANT_CORE, LEGACY_QUANT_CORE)
    return (PACKAGED_QUANT_CORE,)


def _load_status(module_name: str, required: tuple[str, ...]) -> NativeModuleStatus:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return NativeModuleStatus(
            module_name=module_name,
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    missing = tuple(name for name in required if not hasattr(module, name))
    return NativeModuleStatus(
        module_name=module_name,
        available=True,
        source_path=str(getattr(module, "__file__", "")) or None,
        missing_features=missing,
    )


def _prepend_existing_paths(paths: Iterable[str | Path]) -> None:
    changed = False
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            changed = True
    if changed:
        importlib.invalidate_caches()


def _format_status(status: NativeModuleStatus) -> str:
    if status.ok:
        return f"{status.module_name} OK"
    if status.available and status.missing_features:
        return f"{status.module_name} missing {', '.join(status.missing_features)}"
    return f"{status.module_name} unavailable ({status.error or 'unknown error'})"
