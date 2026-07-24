"""Discovery and loading for private strategy risk-overlay recipes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[4]
PRIVATE_STRATEGY_RISK_OVERLAY_ROOT = (
    REPO_ROOT / "departments" / "research" / "strategy_overlays"
)


def iter_strategy_risk_overlay_files() -> tuple[Path, ...]:
    if not PRIVATE_STRATEGY_RISK_OVERLAY_ROOT.exists():
        return ()
    return tuple(sorted(PRIVATE_STRATEGY_RISK_OVERLAY_ROOT.glob("ovl_*.py")))


def resolve_strategy_risk_overlay_path(overlay_name: str) -> Path:
    stem = Path(overlay_name).stem
    for path in iter_strategy_risk_overlay_files():
        if path.stem == stem:
            return path
    raise ModuleNotFoundError(
        f"Could not resolve strategy risk overlay {stem!r} in "
        f"{PRIVATE_STRATEGY_RISK_OVERLAY_ROOT}"
    )


def load_strategy_risk_overlay_module(overlay_name: str) -> ModuleType:
    path = resolve_strategy_risk_overlay_path(overlay_name)
    spec = importlib.util.spec_from_file_location(
        f"oqp_private_strategy_risk_overlay_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy risk overlay from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "OVERLAY_PARAMETERS", None) is not None:
        from oqp.optimization.parameter_spaces import (
            resolve_component_parameter_schema,
        )

        resolve_component_parameter_schema(
            module,
            component_type="risk_overlay",
            declaration_name="OVERLAY_PARAMETERS",
        )
    return module


__all__ = [
    "PRIVATE_STRATEGY_RISK_OVERLAY_ROOT",
    "REPO_ROOT",
    "iter_strategy_risk_overlay_files",
    "load_strategy_risk_overlay_module",
    "resolve_strategy_risk_overlay_path",
]
