"""Native loader for tick-pulse acceleration functions."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from oqp.native import load_quant_core


TICK_PULSE_FEATURE_FUNCTIONS = (
    "compute_tick_pulse_features_core",
    "compute_tick_rtv_pipeline",
    "compute_tick_heuristic_pipeline",
)


def load_tick_pulse_core(
    *,
    required_features: tuple[str, ...] = TICK_PULSE_FEATURE_FUNCTIONS,
    legacy_paths: tuple[str | Path, ...] = (),
) -> ModuleType:
    return load_quant_core(required_features, legacy_paths=legacy_paths)
