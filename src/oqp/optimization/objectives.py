"""Objective extraction for single and multi-objective optimization studies."""

from __future__ import annotations

import math
from typing import Mapping

from oqp.optimization.contracts import ObjectiveSpec


def extract_objective_values(
    metrics: Mapping[str, float],
    objectives: tuple[ObjectiveSpec, ...],
) -> tuple[float, ...]:
    values: list[float] = []
    for objective in objectives:
        if objective.metric not in metrics:
            raise ValueError(
                f"Trial metrics missing objective {objective.metric!r}"
            )
        value = float(metrics[objective.metric])
        if not math.isfinite(value):
            raise ValueError(
                f"Objective {objective.metric!r} must be finite, got {value!r}"
            )
        values.append(value)
    return tuple(values)


__all__ = ["extract_objective_values"]
