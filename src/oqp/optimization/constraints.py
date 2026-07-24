"""Constraint evaluation and Optuna-compatible violation values."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from oqp.optimization.contracts import ConstraintSpec


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    name: str
    metric: str
    value: float
    threshold: float
    operator: str
    violation: float
    satisfied: bool
    hard: bool


def evaluate_constraints(
    metrics: Mapping[str, float],
    constraints: tuple[ConstraintSpec, ...],
) -> tuple[ConstraintEvaluation, ...]:
    results: list[ConstraintEvaluation] = []
    for constraint in constraints:
        if constraint.metric not in metrics:
            raise ValueError(
                f"Trial metrics missing constraint {constraint.metric!r}"
            )
        value = float(metrics[constraint.metric])
        if not math.isfinite(value):
            raise ValueError(
                f"Constraint metric {constraint.metric!r} must be finite"
            )
        violation = _violation(value, constraint)
        results.append(
            ConstraintEvaluation(
                name=constraint.name,
                metric=constraint.metric,
                value=value,
                threshold=float(constraint.threshold),
                operator=constraint.operator,
                violation=float(violation),
                satisfied=violation <= 0.0,
                hard=constraint.hard,
            )
        )
    return tuple(results)


def hard_constraints_satisfied(
    evaluations: tuple[ConstraintEvaluation, ...],
) -> bool:
    return all(item.satisfied for item in evaluations if item.hard)


def optuna_constraint_values(
    evaluations: tuple[ConstraintEvaluation, ...],
) -> tuple[float, ...]:
    return tuple(item.violation for item in evaluations)


def _violation(value: float, constraint: ConstraintSpec) -> float:
    threshold = float(constraint.threshold)
    if constraint.operator == "<=":
        return value - threshold
    if constraint.operator == "<":
        return value - threshold if value >= threshold else -abs(threshold - value)
    if constraint.operator == ">=":
        return threshold - value
    if constraint.operator == ">":
        return threshold - value if value <= threshold else -abs(value - threshold)
    return abs(value - threshold) - 1e-12


__all__ = [
    "ConstraintEvaluation",
    "evaluate_constraints",
    "hard_constraints_satisfied",
    "optuna_constraint_values",
]
