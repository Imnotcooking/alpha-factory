"""Out-of-sample model and feature evaluation utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    name: ("oos_mda", name)
    for name in (
        "PurgedMDAConfig",
        "build_purged_time_folds",
        "compute_oos_mda",
        "default_xgb_regressor",
        "rank_ic_score",
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
