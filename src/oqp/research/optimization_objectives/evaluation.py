"""Compatibility checks and diagnostics for Phase 9 objectives."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Mapping

from oqp.optimization import evaluate_constraints, hard_constraints_satisfied
from oqp.research.optimization_objectives.contracts import (
    OptimizationObjectiveProfile,
)

if TYPE_CHECKING:
    from oqp.research.optional_optimization.contracts import Phase8ExperimentSpec


def validate_phase8_objective_profile(
    spec: "Phase8ExperimentSpec",
    profile: OptimizationObjectiveProfile,
) -> None:
    if spec.objective_profile_id != profile.profile_id:
        raise ValueError("Phase 8 study resolved the wrong objective profile")
    if spec.objective_profile_fingerprint != profile.fingerprint:
        raise ValueError("the frozen Phase 9 objective profile fingerprint changed")
    if spec.layer.value != profile.layer:
        raise ValueError(
            f"objective profile {profile.profile_id} belongs to "
            f"{profile.layer}, not {spec.layer.value}"
        )
    expected_objectives = tuple(
        value.to_phase8_objective() for value in profile.objectives
    )
    if spec.objectives != expected_objectives:
        raise ValueError("study objectives differ from the frozen Phase 9 profile")
    if spec.selection_priority != profile.selection_priority:
        raise ValueError("study selection priority differs from its objective profile")
    if spec.constraints != profile.constraints:
        raise ValueError("study constraints differ from its objective profile")
    component_ids = tuple(spec.frozen_component_fingerprints)
    for requirement in profile.upstream_requirements:
        count = sum(
            any(component_id.startswith(prefix) for prefix in requirement.accepted_prefixes)
            for component_id in component_ids
        )
        if count < requirement.minimum_count:
            prefixes = ", ".join(requirement.accepted_prefixes)
            raise ValueError(
                f"{profile.profile_id} requires {requirement.minimum_count} frozen "
                f"upstream component(s) with prefix {prefixes}"
            )


def evaluate_objective_profile_metrics(
    profile: OptimizationObjectiveProfile,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    objectives = []
    for objective in profile.objectives:
        if objective.posterior_metric not in metrics:
            raise ValueError(
                f"metrics are missing {objective.posterior_metric!r}"
            )
        objectives.append(
            {
                "name": objective.name,
                "metric": objective.metric,
                "direction": objective.direction.value,
                "raw_fold_mean": metrics.get(f"raw__{objective.metric}"),
                "posterior_mean": float(metrics[objective.posterior_metric]),
                "posterior_std": metrics.get(
                    f"posterior_std__{objective.metric}"
                ),
            }
        )
    constraints = evaluate_constraints(metrics, profile.constraints)
    return {
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "layer": profile.layer,
        "objectives": objectives,
        "constraints": [asdict(value) for value in constraints],
        "hard_constraints_satisfied": hard_constraints_satisfied(constraints),
    }


__all__ = [
    "evaluate_objective_profile_metrics",
    "validate_phase8_objective_profile",
]
