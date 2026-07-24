"""Readiness audit and review artifacts for Phase 9 objective profiles."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.research.optimization_objectives.contracts import VALID_PROFILE_LAYERS
from oqp.research.optimization_objectives.registry import (
    DEFAULT_OBJECTIVE_REGISTRY,
    OptimizationObjectiveRegistry,
)


PHASE9_SCHEMA_VERSION = 1


def audit_optimization_objectives(
    registry_path: str | Path = DEFAULT_OBJECTIVE_REGISTRY,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    registry = OptimizationObjectiveRegistry.load(registry_path)
    profiles = pd.DataFrame(registry.inventory())
    objective_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    upstream_rows: list[dict[str, Any]] = []

    for profile in registry.profiles.values():
        for priority, objective_name in enumerate(profile.selection_priority, start=1):
            objective = next(
                value for value in profile.objectives if value.name == objective_name
            )
            objective_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "layer": profile.layer,
                    "priority": priority,
                    "objective": objective.name,
                    "metric": objective.metric,
                    "direction": objective.direction.value,
                    "prior_mean": objective.prior_mean,
                    "prior_std": objective.prior_std,
                    "noise_floor": objective.noise_floor,
                }
            )
        for constraint in profile.constraints:
            constraint_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "layer": profile.layer,
                    **asdict(constraint),
                }
            )
        for requirement in profile.upstream_requirements:
            upstream_rows.append(
                {
                    "profile_id": profile.profile_id,
                    "layer": profile.layer,
                    "requirement": requirement.name,
                    "accepted_prefixes": ", ".join(
                        requirement.accepted_prefixes
                    ),
                    "minimum_count": requirement.minimum_count,
                }
            )

    objectives = pd.DataFrame(objective_rows)
    constraints = pd.DataFrame(constraint_rows)
    upstream = pd.DataFrame(upstream_rows)
    active_profiles = profiles.loc[profiles["status"].eq("active")]
    covered_layers = set(active_profiles["layer"].astype(str))
    missing_layers = sorted(VALID_PROFILE_LAYERS - covered_layers)
    summary = {
        "schema_version": PHASE9_SCHEMA_VERSION,
        "phase": "Phase 9: Optimisation Objectives",
        "status": "active" if not missing_layers else "blocked",
        "registry_id": registry.registry_id,
        "registry_schema_version": registry.schema_version,
        "registry_path": str(registry.source_path),
        "declared_profiles": len(profiles),
        "active_profiles": len(active_profiles),
        "covered_layers": sorted(covered_layers),
        "missing_required_layers": missing_layers,
        "objective_count": len(objectives),
        "hard_constraint_count": int(
            constraints["hard"].sum() if not constraints.empty else 0
        ),
        "overlay_profile_available": False,
        "universal_score_permitted": False,
        "selection_method": "feasible Pareto set then frozen lexicographic priority",
        "boundary": (
            "Each study uses the objective profile for its mutable layer. "
            "Upstream alpha components remain frozen, and objectives are not "
            "combined into one universal weighted score."
        ),
    }
    return summary, profiles, objectives, constraints, upstream


def write_optimization_objective_readiness(
    summary: dict[str, Any],
    profiles: pd.DataFrame,
    objectives: pd.DataFrame,
    constraints: pd.DataFrame,
    upstream: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "readiness.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    profiles.to_csv(destination / "profiles.csv", index=False)
    objectives.to_csv(destination / "objectives.csv", index=False)
    constraints.to_csv(destination / "constraints.csv", index=False)
    upstream.to_csv(destination / "upstream_requirements.csv", index=False)
    return destination


__all__ = [
    "PHASE9_SCHEMA_VERSION",
    "audit_optimization_objectives",
    "write_optimization_objective_readiness",
]
