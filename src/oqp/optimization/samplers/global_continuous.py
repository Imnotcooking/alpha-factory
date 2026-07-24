"""SciPy global optimizers for bounded numerical black-box problems."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from scipy.optimize import differential_evolution, dual_annealing

from oqp.optimization.parameter_spaces import (
    ComponentParameterSchema,
    resolve_component_parameter_values,
)


@dataclass(frozen=True, slots=True)
class GlobalContinuousResult:
    method: str
    parameters: dict[str, Any]
    objective_value: float
    evaluations: int
    success: bool
    message: str


def solve_global_continuous(
    objective: Callable[[dict[str, Any]], float],
    schema: ComponentParameterSchema,
    *,
    method: str = "differential_evolution",
    direction: str = "maximize",
    seed: int = 42,
    max_evaluations: int = 500,
) -> GlobalContinuousResult:
    tunable = [spec for spec in schema.parameters if spec.tunable]
    if not tunable:
        raise ValueError("Global optimization requires tunable parameters")
    unsupported = [
        spec.name
        for spec in tunable
        if spec.parameter_type not in {"int", "float"}
    ]
    if unsupported:
        raise ValueError(
            "Global continuous optimization supports only numerical parameters: "
            + ", ".join(unsupported)
        )
    direction = str(direction).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be maximize or minimize")
    sign = -1.0 if direction == "maximize" else 1.0
    bounds = [
        (
            math.log(float(spec.low)) if spec.log else float(spec.low),
            math.log(float(spec.high)) if spec.log else float(spec.high),
        )
        for spec in tunable
    ]
    evaluations = 0

    def wrapped(values) -> float:
        nonlocal evaluations
        evaluations += 1
        overrides: dict[str, Any] = {}
        for spec, value in zip(tunable, values, strict=True):
            decoded = math.exp(float(value)) if spec.log else float(value)
            overrides[spec.name] = _project_numeric_value(spec, decoded)
        parameters = resolve_component_parameter_values(
            schema, overrides, enforce_search_bounds=True
        )
        return sign * float(objective(parameters))

    method = str(method).strip().lower()
    if method == "differential_evolution":
        dimension = max(len(tunable), 1)
        population_size = 5
        max_iterations = max(
            1,
            int(max_evaluations) // (population_size * dimension) - 1,
        )
        result = differential_evolution(
            wrapped,
            bounds,
            seed=int(seed),
            popsize=population_size,
            maxiter=max_iterations,
            polish=False,
            workers=1,
        )
    elif method == "dual_annealing":
        result = dual_annealing(
            wrapped,
            bounds,
            seed=int(seed),
            maxfun=int(max_evaluations),
            no_local_search=True,
        )
    else:
        raise ValueError(
            "method must be differential_evolution or dual_annealing"
        )
    best_overrides = {
        spec.name: _project_numeric_value(
            spec,
            math.exp(float(value)) if spec.log else float(value),
        )
        for spec, value in zip(tunable, result.x, strict=True)
    }
    return GlobalContinuousResult(
        method=method,
        parameters=resolve_component_parameter_values(
            schema, best_overrides, enforce_search_bounds=True
        ),
        objective_value=float(sign * result.fun),
        evaluations=int(evaluations),
        success=bool(result.success),
        message=str(result.message),
    )


def _project_numeric_value(spec, value: float) -> int | float:
    low = float(spec.low)
    high = float(spec.high)
    projected = min(max(float(value), low), high)
    if spec.step is not None:
        step = float(spec.step)
        step_count = round((projected - low) / step)
        max_step_count = math.floor((high - low) / step + 1e-12)
        step_count = min(max(step_count, 0), max_step_count)
        projected = low + step_count * step
    if spec.parameter_type == "int":
        return int(round(projected))
    return float(projected)


__all__ = ["GlobalContinuousResult", "solve_global_continuous"]
