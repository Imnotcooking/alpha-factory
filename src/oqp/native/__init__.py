"""Native acceleration helpers for OQP.

The public Python surface should import through this package instead of
importing pybind11 extensions directly. That keeps research code portable
between the packaged native kernel and temporary legacy lab builds.
"""

from oqp.native.backend import (
    NativeModuleStatus,
    QuantCoreUnavailable,
    load_quant_core,
    quant_core_status,
)

__all__ = [
    "NativeModuleStatus",
    "QuantCoreUnavailable",
    "load_quant_core",
    "quant_core_status",
]
