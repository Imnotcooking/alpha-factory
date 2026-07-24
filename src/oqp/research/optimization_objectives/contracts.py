"""Versioned layer-specific objective profiles for Phase 9."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from oqp.optimization import (
    ConstraintSpec,
    OptimizationDirection,
    stable_optimization_hash,
)


VALID_PROFILE_LAYERS = {"factor", "sleeve", "router", "allocator"}


@dataclass(frozen=True, slots=True)
class OptimizationProfileObjectiveSpec:
    name: str
    metric: str
    direction: OptimizationDirection | str
    prior_mean: float
    prior_std: float
    noise_floor: float

    def __post_init__(self) -> None:
        direction = (
            self.direction
            if isinstance(self.direction, OptimizationDirection)
            else OptimizationDirection(str(self.direction))
        )
        if not str(self.name).strip() or not str(self.metric).strip():
            raise ValueError("profile objective name and metric are required")
        if float(self.prior_std) <= 0.0 or float(self.noise_floor) <= 0.0:
            raise ValueError("profile prior_std and noise_floor must be positive")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "metric", str(self.metric).strip())
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "prior_mean", float(self.prior_mean))
        object.__setattr__(self, "prior_std", float(self.prior_std))
        object.__setattr__(self, "noise_floor", float(self.noise_floor))

    @property
    def posterior_metric(self) -> str:
        return f"posterior__{self.metric}"

    def to_phase8_objective(self):
        from oqp.research.optional_optimization.contracts import Phase8ObjectiveSpec

        return Phase8ObjectiveSpec(
            name=self.name,
            metric=self.metric,
            direction=self.direction,
            prior_mean=self.prior_mean,
            prior_std=self.prior_std,
            noise_floor=self.noise_floor,
        )


@dataclass(frozen=True, slots=True)
class UpstreamEvidenceRequirement:
    name: str
    accepted_prefixes: tuple[str, ...]
    minimum_count: int = 1

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        prefixes = tuple(str(value).strip() for value in self.accepted_prefixes)
        if not name or not prefixes or any(not value for value in prefixes):
            raise ValueError("upstream requirement name and prefixes are required")
        if int(self.minimum_count) < 1:
            raise ValueError("upstream requirement minimum_count must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "accepted_prefixes", prefixes)
        object.__setattr__(self, "minimum_count", int(self.minimum_count))


@dataclass(frozen=True, slots=True)
class OptimizationObjectiveProfile:
    profile_id: str
    layer: str
    economic_question: str
    objectives: tuple[OptimizationProfileObjectiveSpec, ...]
    selection_priority: tuple[str, ...]
    constraints: tuple[ConstraintSpec, ...] = ()
    upstream_requirements: tuple[UpstreamEvidenceRequirement, ...] = ()
    status: str = "active"

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        question = str(self.economic_question).strip()
        layer = str(self.layer).strip().lower()
        if layer not in VALID_PROFILE_LAYERS:
            raise ValueError(f"unsupported Phase 9 objective layer: {layer}")
        objectives = tuple(self.objectives)
        priorities = tuple(str(value).strip() for value in self.selection_priority)
        names = tuple(value.name for value in objectives)
        if not profile_id or not question:
            raise ValueError("objective profile ID and economic question are required")
        if not objectives or len(set(names)) != len(names):
            raise ValueError("objective profiles require unique objectives")
        if priorities != tuple(dict.fromkeys(priorities)) or set(priorities) != set(names):
            raise ValueError("selection priority must list each objective exactly once")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "economic_question", question)
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "selection_priority", priorities)
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(
            self, "upstream_requirements", tuple(self.upstream_requirements)
        )
        object.__setattr__(self, "status", str(self.status).strip().lower())

    @property
    def fingerprint(self) -> str:
        return stable_optimization_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "layer": self.layer,
            "economic_question": self.economic_question,
            "objectives": [
                {**asdict(value), "direction": value.direction.value}
                for value in self.objectives
            ],
            "selection_priority": list(self.selection_priority),
            "constraints": [asdict(value) for value in self.constraints],
            "upstream_requirements": [
                asdict(value) for value in self.upstream_requirements
            ],
            "status": self.status,
        }

    @classmethod
    def from_mapping(
        cls, profile_id: str, payload: Mapping[str, Any]
    ) -> "OptimizationObjectiveProfile":
        return cls(
            profile_id=profile_id,
            layer=str(payload.get("layer") or ""),
            economic_question=str(payload.get("economic_question") or ""),
            objectives=tuple(
                OptimizationProfileObjectiveSpec(**dict(value))
                for value in (payload.get("objectives") or ())
            ),
            selection_priority=tuple(payload.get("selection_priority") or ()),
            constraints=tuple(
                ConstraintSpec(**dict(value))
                for value in (payload.get("constraints") or ())
            ),
            upstream_requirements=tuple(
                UpstreamEvidenceRequirement(
                    name=str(value.get("name") or ""),
                    accepted_prefixes=tuple(value.get("accepted_prefixes") or ()),
                    minimum_count=int(value.get("minimum_count", 1)),
                )
                for value in (payload.get("upstream_requirements") or ())
            ),
            status=str(payload.get("status") or "active"),
        )


__all__ = [
    "OptimizationObjectiveProfile",
    "OptimizationProfileObjectiveSpec",
    "UpstreamEvidenceRequirement",
    "VALID_PROFILE_LAYERS",
]
