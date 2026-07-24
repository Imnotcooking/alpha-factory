"""Immutable contracts shared by optimization engines and research components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


class OptimizationPurpose(str, Enum):
    FACTOR_PARAMETER = "factor_parameter"
    SLEEVE_PARAMETER = "sleeve_parameter"
    ROUTER_PARAMETER = "router_parameter"
    ALLOCATOR_PARAMETER = "allocator_parameter"
    OVERLAY_PARAMETER = "overlay_parameter"
    MODEL_HYPERPARAMETER = "model_hyperparameter"
    MODEL_WEIGHT_TRAINING = "model_weight_training"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    UNIVERSE_SELECTION = "universe_selection"


class OptimizationDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    name: str
    metric: str
    direction: OptimizationDirection | str = OptimizationDirection.MAXIMIZE

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        metric = str(self.metric).strip()
        if not name or not metric:
            raise ValueError("objective name and metric cannot be empty")
        direction = (
            self.direction
            if isinstance(self.direction, OptimizationDirection)
            else OptimizationDirection(str(self.direction))
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    name: str
    metric: str
    operator: str
    threshold: float
    hard: bool = True

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.metric).strip():
            raise ValueError("constraint name and metric cannot be empty")
        if self.operator not in {"<=", ">=", "<", ">", "=="}:
            raise ValueError("constraint operator must be <=, >=, <, >, or ==")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("constraint threshold must be finite")


@dataclass(frozen=True, slots=True)
class SearchBudget:
    max_trials: int
    timeout_seconds: int | None = None
    max_grid_combinations: int = 500
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if int(self.max_trials) < 1:
            raise ValueError("max_trials must be at least 1")
        if self.timeout_seconds is not None and int(self.timeout_seconds) < 1:
            raise ValueError("timeout_seconds must be positive or null")
        if int(self.max_grid_combinations) < 1:
            raise ValueError("max_grid_combinations must be at least 1")
        if int(self.n_jobs) < 1:
            raise ValueError("n_jobs must be at least 1")
        object.__setattr__(self, "max_trials", int(self.max_trials))
        object.__setattr__(
            self,
            "timeout_seconds",
            None if self.timeout_seconds is None else int(self.timeout_seconds),
        )
        object.__setattr__(
            self, "max_grid_combinations", int(self.max_grid_combinations)
        )
        object.__setattr__(self, "n_jobs", int(self.n_jobs))


@dataclass(frozen=True, slots=True)
class FrozenResearchInputs:
    dataset_fingerprint: str
    universe_fingerprint: str = ""
    liquidity_policy_fingerprint: str = ""
    temporal_policy_fingerprint: str = ""
    transaction_cost_profile_fingerprint: str = ""
    holdout_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not str(self.dataset_fingerprint).strip():
            raise ValueError("dataset_fingerprint is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OptimizationStudySpec:
    study_id: str
    purpose: OptimizationPurpose | str
    component_id: str
    sampler_id: str
    objectives: tuple[ObjectiveSpec, ...]
    frozen_inputs: FrozenResearchInputs
    budget: SearchBudget
    constraints: tuple[ConstraintSpec, ...] = ()
    seed: int = 42
    holdout_locked: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        study_id = str(self.study_id).strip()
        component_id = str(self.component_id).strip()
        sampler_id = str(self.sampler_id).strip().lower()
        if not study_id or not component_id or not sampler_id:
            raise ValueError("study_id, component_id, and sampler_id are required")
        objectives = tuple(self.objectives)
        if not objectives:
            raise ValueError("at least one objective is required")
        if len({objective.name for objective in objectives}) != len(objectives):
            raise ValueError("objective names must be unique")
        if not self.holdout_locked:
            raise ValueError("optimization studies must keep the final holdout locked")
        object.__setattr__(self, "study_id", study_id)
        purpose = (
            self.purpose
            if isinstance(self.purpose, OptimizationPurpose)
            else OptimizationPurpose(str(self.purpose))
        )
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "sampler_id", sampler_id)
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def directions(self) -> tuple[str, ...]:
        return tuple(objective.direction.value for objective in self.objectives)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "purpose": self.purpose.value,
            "component_id": self.component_id,
            "sampler_id": self.sampler_id,
            "objectives": [
                {
                    "name": objective.name,
                    "metric": objective.metric,
                    "direction": objective.direction.value,
                }
                for objective in self.objectives
            ],
            "constraints": [asdict(constraint) for constraint in self.constraints],
            "frozen_inputs": self.frozen_inputs.to_dict(),
            "budget": asdict(self.budget),
            "seed": self.seed,
            "holdout_locked": self.holdout_locked,
            "metadata": dict(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        return stable_optimization_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class OptimizationEvaluationContext:
    study_id: str
    trial_number: int
    purpose: OptimizationPurpose
    component_id: str
    development_dataset_fingerprint: str
    holdout_locked: bool = True

    def require_development_only(self) -> None:
        if not self.holdout_locked:
            raise RuntimeError("optimization context unexpectedly unlocked the holdout")


@dataclass(frozen=True, slots=True)
class TrialEvaluation:
    metrics: Mapping[str, float]
    fold_metrics: tuple[Mapping[str, Any], ...] = ()
    artifacts: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if not metrics:
            raise ValueError("trial evaluation metrics cannot be empty")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(
            self, "fold_metrics", tuple(dict(item) for item in self.fold_metrics)
        )
        object.__setattr__(self, "artifacts", dict(self.artifacts))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class OptimizationCandidate:
    trial_number: int
    parameters: Mapping[str, Any]
    objective_values: tuple[float, ...]
    metrics: Mapping[str, float]
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "parameters": dict(self.parameters),
            "objective_values": list(self.objective_values),
            "metrics": dict(self.metrics),
            "feasible": self.feasible,
        }


@dataclass(frozen=True, slots=True)
class OptimizationStudyResult:
    study_id: str
    study_fingerprint: str
    parameter_schema_fingerprint: str
    sampler_id: str
    trial_count: int
    candidates: tuple[OptimizationCandidate, ...]
    diagnostics: Mapping[str, Any]
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "study_fingerprint": self.study_fingerprint,
            "parameter_schema_fingerprint": self.parameter_schema_fingerprint,
            "sampler_id": self.sampler_id,
            "trial_count": self.trial_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "diagnostics": dict(self.diagnostics),
            "artifact_path": self.artifact_path,
        }


def stable_optimization_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ConstraintSpec",
    "FrozenResearchInputs",
    "ObjectiveSpec",
    "OptimizationCandidate",
    "OptimizationDirection",
    "OptimizationEvaluationContext",
    "OptimizationPurpose",
    "OptimizationStudyResult",
    "OptimizationStudySpec",
    "SearchBudget",
    "TrialEvaluation",
    "stable_optimization_hash",
]
