"""Reusable adaptive state-space estimators and diagnostics.

The package initializer is deliberately lazy so importing the ML umbrella does
not initialize pandas or NumPy until a state-space component is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "StateSpaceArtifact": ("base", "StateSpaceArtifact"),
    "StateSpaceFilter": ("base", "StateSpaceFilter"),
    "StateSpaceSchema": ("base", "StateSpaceSchema"),
    "dataclass_to_dict": ("base", "dataclass_to_dict"),
    "coefficient_columns": ("diagnostics", "coefficient_columns"),
    "summarize_dual_kalman_output": (
        "diagnostics",
        "summarize_dual_kalman_output",
    ),
    "DualKalmanRegression": (
        "dual_kalman_regression",
        "DualKalmanRegression",
    ),
    "DualKalmanRegressionConfig": (
        "dual_kalman_regression",
        "DualKalmanRegressionConfig",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
