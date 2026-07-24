"""Discovery and loading for private router recipes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[4]
PRIVATE_ROUTER_ROOT = REPO_ROOT / "departments" / "research" / "routers"


def iter_router_files() -> tuple[Path, ...]:
    if not PRIVATE_ROUTER_ROOT.exists():
        return ()
    return tuple(sorted(PRIVATE_ROUTER_ROOT.glob("rtr_*.py")))


def resolve_router_path(router_name: str) -> Path:
    stem = Path(router_name).stem
    for path in iter_router_files():
        if path.stem == stem:
            return path
    raise ModuleNotFoundError(
        f"Could not resolve router module {stem!r} in {PRIVATE_ROUTER_ROOT}"
    )


def load_router_module(router_name: str) -> ModuleType:
    path = resolve_router_path(router_name)
    spec = importlib.util.spec_from_file_location(
        f"oqp_private_router_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load router module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "ROUTER_PARAMETERS", None) is not None:
        from oqp.optimization.parameter_spaces import (
            resolve_component_parameter_schema,
        )

        resolve_component_parameter_schema(module, component_type="router")
    return module


__all__ = [
    "PRIVATE_ROUTER_ROOT",
    "REPO_ROOT",
    "iter_router_files",
    "load_router_module",
    "resolve_router_path",
]
