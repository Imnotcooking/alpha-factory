"""Optimizer adapters and robustness diagnostics for factor parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping

from oqp.research.parameter_schema import (
    FactorParameterSchema,
    FactorParameterSpec,
    resolve_factor_parameter_schema,
    resolve_parameter_values,
)


OPTIMIZATION_EVIDENCE_REQUIREMENTS = (
    "validation_ic",
    "validation_rank_ic",
    "validation_icir",
    "walk_forward_fold_results",
    "boundary_diagnostics",
    "neighborhood_diagnostics",
    "turnover_and_costs",
    "untouched_holdout_result",
    "economic_interpretation_review",
)


@dataclass(frozen=True, slots=True)
class ParameterBoundaryDiagnostic:
    parameter: str
    value: int | float
    low: int | float
    high: int | float
    normalized_position: float
    at_lower_boundary: bool
    at_upper_boundary: bool
    near_lower_boundary: bool
    near_upper_boundary: bool

    @property
    def at_search_boundary(self) -> bool:
        return self.at_lower_boundary or self.at_upper_boundary

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["at_search_boundary"] = self.at_search_boundary
        return payload


@dataclass(frozen=True, slots=True)
class OptimizationObservation:
    parameters: Mapping[str, Any]
    objective: float


@dataclass(frozen=True, slots=True)
class ParameterSurfaceDiagnostic:
    direction: str
    observation_count: int
    best_objective: float
    near_best_tolerance: float
    near_best_count: int
    near_best_share: float
    neighbor_count: int
    near_best_neighbor_count: int
    neighborhood_status: str
    broad_plateau_observed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def suggest_factor_parameters(
    trial: Any,
    schema_or_module: FactorParameterSchema | Any,
) -> dict[str, Any]:
    """Ask an Optuna-compatible trial for tunable values declared by a factor."""

    schema = _coerce_schema(schema_or_module)
    suggestions: dict[str, Any] = {}
    for spec in schema.parameters:
        if not spec.tunable:
            continue
        if spec.parameter_type == "int":
            kwargs: dict[str, Any] = {}
            if spec.step is not None:
                kwargs["step"] = int(spec.step)
            if spec.log:
                kwargs["log"] = True
            suggestions[spec.name] = trial.suggest_int(
                spec.name,
                int(spec.low),
                int(spec.high),
                **kwargs,
            )
        elif spec.parameter_type == "float":
            kwargs = {}
            if spec.step is not None:
                kwargs["step"] = float(spec.step)
            if spec.log:
                kwargs["log"] = True
            suggestions[spec.name] = trial.suggest_float(
                spec.name,
                float(spec.low),
                float(spec.high),
                **kwargs,
            )
        elif spec.parameter_type == "bool":
            suggestions[spec.name] = trial.suggest_categorical(
                spec.name,
                [False, True],
            )
        elif spec.parameter_type == "categorical":
            suggestions[spec.name] = trial.suggest_categorical(
                spec.name,
                list(spec.choices),
            )
        else:  # pragma: no cover - schema validation prevents this branch
            raise ValueError(
                f"Unsupported tunable type {spec.parameter_type!r} for {spec.name}"
            )
    return resolve_parameter_values(
        schema,
        suggestions,
        enforce_search_bounds=True,
    )


def diagnose_parameter_boundaries(
    best_parameters: Mapping[str, Any],
    schema_or_module: FactorParameterSchema | Any,
    *,
    near_boundary_fraction: float = 0.05,
) -> tuple[ParameterBoundaryDiagnostic, ...]:
    """Flag exact and near-boundary numerical optima without judging performance."""

    schema = _coerce_schema(schema_or_module)
    if near_boundary_fraction < 0:
        raise ValueError("near_boundary_fraction cannot be negative")
    diagnostics: list[ParameterBoundaryDiagnostic] = []
    for spec in schema.parameters:
        if not spec.tunable or spec.parameter_type not in {"int", "float"}:
            continue
        value = float(best_parameters.get(spec.name, spec.default))
        low = float(spec.low)
        high = float(spec.high)
        width = high - low
        exact_tolerance = max(abs(width) * 1e-12, 1e-15)
        near_distance = (
            float(spec.step)
            if spec.step is not None
            else width * float(near_boundary_fraction)
        )
        diagnostics.append(
            ParameterBoundaryDiagnostic(
                parameter=spec.name,
                value=_preserve_numeric_type(value, spec),
                low=_preserve_numeric_type(low, spec),
                high=_preserve_numeric_type(high, spec),
                normalized_position=(value - low) / width,
                at_lower_boundary=math.isclose(
                    value, low, rel_tol=0.0, abs_tol=exact_tolerance
                ),
                at_upper_boundary=math.isclose(
                    value, high, rel_tol=0.0, abs_tol=exact_tolerance
                ),
                near_lower_boundary=value - low <= near_distance + exact_tolerance,
                near_upper_boundary=high - value <= near_distance + exact_tolerance,
            )
        )
    return tuple(diagnostics)


def diagnose_parameter_surface(
    observations: Iterable[OptimizationObservation | Mapping[str, Any] | Any],
    schema_or_module: FactorParameterSchema | Any,
    *,
    direction: str = "maximize",
    relative_tolerance: float = 0.05,
    absolute_tolerance: float = 0.0,
    neighborhood_radius: float = 1.0,
) -> ParameterSurfaceDiagnostic:
    """Check whether the best result is supported by nearby parameter values."""

    schema = _coerce_schema(schema_or_module)
    direction = str(direction).strip().lower()
    if direction not in {"maximize", "minimize"}:
        raise ValueError("direction must be 'maximize' or 'minimize'")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("objective tolerances cannot be negative")
    if neighborhood_radius <= 0:
        raise ValueError("neighborhood_radius must be positive")

    records = tuple(_coerce_observation(item) for item in observations)
    records = tuple(record for record in records if math.isfinite(record.objective))
    if not records:
        raise ValueError("At least one finite optimization observation is required")
    best_index = (
        max(range(len(records)), key=lambda index: records[index].objective)
        if direction == "maximize"
        else min(range(len(records)), key=lambda index: records[index].objective)
    )
    best = records[best_index]
    tolerance = max(
        abs(best.objective) * float(relative_tolerance),
        float(absolute_tolerance),
    )

    def near_best(objective: float) -> bool:
        if direction == "maximize":
            return objective >= best.objective - tolerance
        return objective <= best.objective + tolerance

    near_best_count = sum(near_best(record.objective) for record in records)
    neighbors = [
        record
        for index, record in enumerate(records)
        if index != best_index
        and _is_local_neighbor(
            record.parameters,
            best.parameters,
            schema,
            radius=float(neighborhood_radius),
        )
    ]
    near_best_neighbors = sum(near_best(record.objective) for record in neighbors)
    if not neighbors:
        neighborhood_status = "insufficient_neighborhood_sampling"
    elif near_best_neighbors:
        neighborhood_status = "supported_by_nearby_values"
    else:
        neighborhood_status = "isolated_optimum"
    broad_plateau = (
        near_best_count >= 3
        and near_best_neighbors > 0
    )
    return ParameterSurfaceDiagnostic(
        direction=direction,
        observation_count=len(records),
        best_objective=float(best.objective),
        near_best_tolerance=float(tolerance),
        near_best_count=int(near_best_count),
        near_best_share=float(near_best_count / len(records)),
        neighbor_count=len(neighbors),
        near_best_neighbor_count=int(near_best_neighbors),
        neighborhood_status=neighborhood_status,
        broad_plateau_observed=bool(broad_plateau),
    )


def _coerce_schema(schema_or_module: FactorParameterSchema | Any) -> FactorParameterSchema:
    if isinstance(schema_or_module, FactorParameterSchema):
        return schema_or_module
    return resolve_factor_parameter_schema(schema_or_module)


def _coerce_observation(item: OptimizationObservation | Mapping[str, Any] | Any) -> OptimizationObservation:
    if isinstance(item, OptimizationObservation):
        return item
    if isinstance(item, Mapping):
        parameters = item.get("parameters", item.get("params"))
        objective = item.get("objective", item.get("value"))
    else:
        parameters = getattr(item, "params", None)
        objective = getattr(item, "value", None)
    if not isinstance(parameters, Mapping) or objective is None:
        raise ValueError(
            "Optimization observations require parameters/params and objective/value"
        )
    return OptimizationObservation(
        parameters=dict(parameters),
        objective=float(objective),
    )


def _is_local_neighbor(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    schema: FactorParameterSchema,
    *,
    radius: float,
) -> bool:
    compared = False
    for spec in schema.parameters:
        if not spec.tunable:
            continue
        candidate_value = candidate.get(spec.name, spec.default)
        reference_value = reference.get(spec.name, spec.default)
        compared = True
        if spec.parameter_type in {"categorical", "bool"}:
            if candidate_value != reference_value:
                return False
            continue
        if not _numeric_values_are_neighbors(
            float(candidate_value),
            float(reference_value),
            spec,
            radius=radius,
        ):
            return False
    return compared


def _numeric_values_are_neighbors(
    candidate: float,
    reference: float,
    spec: FactorParameterSpec,
    *,
    radius: float,
) -> bool:
    if spec.step is not None:
        distance = abs(candidate - reference) / float(spec.step)
    elif spec.log:
        log_width = math.log(float(spec.high)) - math.log(float(spec.low))
        distance = abs(math.log(candidate) - math.log(reference)) / (0.10 * log_width)
    else:
        scale = 0.10 * (float(spec.high) - float(spec.low))
        distance = abs(candidate - reference) / scale
    return distance <= radius + 1e-12


def _preserve_numeric_type(value: float, spec: FactorParameterSpec) -> int | float:
    if spec.parameter_type == "int":
        return int(round(value))
    return float(value)


__all__ = [
    "OPTIMIZATION_EVIDENCE_REQUIREMENTS",
    "OptimizationObservation",
    "ParameterBoundaryDiagnostic",
    "ParameterSurfaceDiagnostic",
    "diagnose_parameter_boundaries",
    "diagnose_parameter_surface",
    "suggest_factor_parameters",
]
