"""Frozen governance contracts for optional Phase 8 optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from oqp.optimization import (
    ConstraintSpec,
    FrozenResearchInputs,
    ObjectiveSpec,
    OptimizationDirection,
    OptimizationPurpose,
    OptimizationStudySpec,
    SearchBudget,
    stable_optimization_hash,
)


PHASE8_SCHEMA_VERSION = 3
PHASE8_SUPPORTED_SAMPLERS = frozenset(
    {
        "bruteforce",
        "cmaes",
        "gp",
        "grid",
        "nsga2",
        "qmc",
        "random",
        "tpe",
    }
)


class OptimizationLayer(str, Enum):
    FACTOR = "factor"
    SLEEVE = "sleeve"
    ROUTER = "router"
    ALLOCATOR = "allocator"
    OVERLAY = "overlay"


LAYER_COMPONENT_TYPES = {
    OptimizationLayer.FACTOR: "factor",
    OptimizationLayer.SLEEVE: "sleeve",
    OptimizationLayer.ROUTER: "router",
    OptimizationLayer.ALLOCATOR: "allocator",
    OptimizationLayer.OVERLAY: "risk_overlay",
}


LAYER_PURPOSES = {
    OptimizationLayer.FACTOR: OptimizationPurpose.FACTOR_PARAMETER,
    OptimizationLayer.SLEEVE: OptimizationPurpose.SLEEVE_PARAMETER,
    OptimizationLayer.ROUTER: OptimizationPurpose.ROUTER_PARAMETER,
    OptimizationLayer.ALLOCATOR: OptimizationPurpose.ALLOCATOR_PARAMETER,
    OptimizationLayer.OVERLAY: OptimizationPurpose.OVERLAY_PARAMETER,
}


@dataclass(frozen=True, slots=True)
class Phase8ObjectiveSpec:
    name: str
    metric: str
    direction: OptimizationDirection | str = OptimizationDirection.MAXIMIZE
    prior_mean: float = 0.0
    prior_std: float = 1.0
    noise_floor: float = 1e-6

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        metric = str(self.metric).strip()
        direction = (
            self.direction
            if isinstance(self.direction, OptimizationDirection)
            else OptimizationDirection(str(self.direction))
        )
        if not name or not metric:
            raise ValueError("objective name and metric cannot be empty")
        if float(self.prior_std) <= 0.0:
            raise ValueError("objective prior_std must be positive")
        if float(self.noise_floor) <= 0.0:
            raise ValueError("objective noise_floor must be positive")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "prior_mean", float(self.prior_mean))
        object.__setattr__(self, "prior_std", float(self.prior_std))
        object.__setattr__(self, "noise_floor", float(self.noise_floor))

    @property
    def posterior_metric(self) -> str:
        return f"posterior__{self.metric}"

    def to_objective_spec(self) -> ObjectiveSpec:
        return ObjectiveSpec(
            name=self.name,
            metric=self.posterior_metric,
            direction=self.direction,
        )


@dataclass(frozen=True, slots=True)
class Phase8FoldConfig:
    fold_count: int = 4
    minimum_training_periods: int = 252
    validation_periods: int = 63
    purge_periods: int = 1
    embargo_periods: int = 5
    date_col: str = "date"

    def __post_init__(self) -> None:
        if int(self.fold_count) < 2:
            raise ValueError("Phase 8 requires at least two inner folds")
        if int(self.minimum_training_periods) < 2:
            raise ValueError("minimum_training_periods must be at least two")
        if int(self.validation_periods) < 1:
            raise ValueError("validation_periods must be positive")
        if int(self.purge_periods) < 0 or int(self.embargo_periods) < 0:
            raise ValueError("purge and embargo periods cannot be negative")
        if not str(self.date_col).strip():
            raise ValueError("date_col cannot be empty")
        for field_name in (
            "fold_count",
            "minimum_training_periods",
            "validation_periods",
            "purge_periods",
            "embargo_periods",
        ):
            object.__setattr__(self, field_name, int(getattr(self, field_name)))
        object.__setattr__(self, "date_col", str(self.date_col).strip())


@dataclass(frozen=True, slots=True)
class Phase8ExperimentSpec:
    study_id: str
    layer: OptimizationLayer | str
    component_id: str
    parameter_schema_fingerprint: str
    objective_profile_id: str
    objective_profile_fingerprint: str
    objectives: tuple[Phase8ObjectiveSpec, ...]
    selection_priority: tuple[str, ...]
    frozen_inputs: FrozenResearchInputs
    budget: SearchBudget
    fold_config: Phase8FoldConfig
    holdout_start: str
    frozen_on: str
    constraints: tuple[ConstraintSpec, ...] = ()
    frozen_component_fingerprints: Mapping[str, str] = field(default_factory=dict)
    sampler_id: str = "tpe"
    seed: int = 42
    enabled: bool = False
    schema_version: int = PHASE8_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "study_id",
            "component_id",
            "parameter_schema_fingerprint",
            "objective_profile_id",
            "objective_profile_fingerprint",
            "holdout_start",
            "frozen_on",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} cannot be empty")
            object.__setattr__(self, field_name, value)
        layer = (
            self.layer
            if isinstance(self.layer, OptimizationLayer)
            else OptimizationLayer(str(self.layer).strip().lower())
        )
        objectives = tuple(self.objectives)
        priorities = tuple(str(value).strip() for value in self.selection_priority)
        if not objectives:
            raise ValueError("Phase 8 requires at least one objective")
        names = tuple(objective.name for objective in objectives)
        if len(set(names)) != len(names):
            raise ValueError("Phase 8 objective names must be unique")
        if not priorities or set(priorities) != set(names):
            raise ValueError(
                "selection_priority must list every objective exactly once"
            )
        sampler_id = str(self.sampler_id).strip().lower()
        if sampler_id not in PHASE8_SUPPORTED_SAMPLERS:
            raise ValueError(
                "Phase 8 sampler must be one of "
                + ", ".join(sorted(PHASE8_SUPPORTED_SAMPLERS))
            )
        if len(objectives) > 1 and sampler_id in {"cmaes", "gp"}:
            raise ValueError(
                f"Phase 8 sampler {sampler_id!r} does not support "
                "multi-objective studies"
            )
        if not str(self.frozen_inputs.holdout_fingerprint).strip():
            raise ValueError("Phase 8 requires a frozen final holdout fingerprint")
        frozen = pd.Timestamp(self.frozen_on).normalize()
        holdout = pd.Timestamp(self.holdout_start).normalize()
        if frozen >= holdout:
            raise ValueError("the Phase 8 protocol must be frozen before holdout")
        frozen_components = {
            str(key).strip(): str(value).strip()
            for key, value in self.frozen_component_fingerprints.items()
        }
        if self.component_id in frozen_components:
            raise ValueError(
                "the mutable component cannot also appear among frozen components"
            )
        if any(not key or not value for key, value in frozen_components.items()):
            raise ValueError("frozen component IDs and fingerprints cannot be empty")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "objectives", objectives)
        object.__setattr__(self, "selection_priority", priorities)
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(
            self, "frozen_component_fingerprints", frozen_components
        )
        object.__setattr__(self, "sampler_id", sampler_id)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "frozen_on", frozen.date().isoformat())
        object.__setattr__(self, "holdout_start", holdout.date().isoformat())

    @property
    def expected_component_type(self) -> str:
        return LAYER_COMPONENT_TYPES[self.layer]

    @property
    def fingerprint(self) -> str:
        return stable_optimization_hash(self.to_dict())

    def to_optimization_study_spec(self) -> OptimizationStudySpec:
        return OptimizationStudySpec(
            study_id=self.study_id,
            purpose=LAYER_PURPOSES[self.layer],
            component_id=self.component_id,
            sampler_id=self.sampler_id,
            objectives=tuple(
                objective.to_objective_spec() for objective in self.objectives
            ),
            frozen_inputs=self.frozen_inputs,
            budget=self.budget,
            constraints=self.constraints,
            seed=self.seed,
            holdout_locked=True,
            metadata={
                "phase": "Phase 8: Optional Optimisation",
                "mutable_layer": self.layer.value,
                "phase8_protocol_fingerprint": self.fingerprint,
                "objective_profile_id": self.objective_profile_id,
                "objective_profile_fingerprint": self.objective_profile_fingerprint,
                "selection_priority": list(self.selection_priority),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "layer": self.layer.value,
            "component_id": self.component_id,
            "parameter_schema_fingerprint": self.parameter_schema_fingerprint,
            "objective_profile_id": self.objective_profile_id,
            "objective_profile_fingerprint": self.objective_profile_fingerprint,
            "objectives": [
                {
                    **asdict(objective),
                    "direction": objective.direction.value,
                }
                for objective in self.objectives
            ],
            "selection_priority": list(self.selection_priority),
            "frozen_inputs": self.frozen_inputs.to_dict(),
            "budget": asdict(self.budget),
            "fold_config": asdict(self.fold_config),
            "holdout_start": self.holdout_start,
            "frozen_on": self.frozen_on,
            "constraints": [asdict(value) for value in self.constraints],
            "frozen_component_fingerprints": dict(
                self.frozen_component_fingerprints
            ),
            "sampler_id": self.sampler_id,
            "seed": self.seed,
            "enabled": self.enabled,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Phase8ExperimentSpec":
        raw = payload.get("optimization", payload)
        if not isinstance(raw, Mapping):
            raise ValueError("Phase 8 configuration must be a mapping")
        mutable_layers = raw.get("mutable_layers")
        if mutable_layers is not None:
            if not isinstance(mutable_layers, (list, tuple)) or len(mutable_layers) != 1:
                raise ValueError("Phase 8 permits exactly one mutable layer")
            declared_layer = mutable_layers[0]
        else:
            declared_layer = raw.get("layer")
        return cls(
            study_id=str(raw.get("study_id") or ""),
            layer=str(declared_layer or ""),
            component_id=str(raw.get("component_id") or ""),
            parameter_schema_fingerprint=str(
                raw.get("parameter_schema_fingerprint") or ""
            ),
            objective_profile_id=str(raw.get("objective_profile_id") or ""),
            objective_profile_fingerprint=str(
                raw.get("objective_profile_fingerprint") or ""
            ),
            objectives=tuple(
                Phase8ObjectiveSpec(**dict(value))
                for value in (raw.get("objectives") or ())
            ),
            selection_priority=tuple(raw.get("selection_priority") or ()),
            frozen_inputs=FrozenResearchInputs(**dict(raw.get("frozen_inputs") or {})),
            budget=SearchBudget(**dict(raw.get("budget") or {})),
            fold_config=Phase8FoldConfig(**dict(raw.get("fold_config") or {})),
            holdout_start=str(raw.get("holdout_start") or ""),
            frozen_on=str(raw.get("frozen_on") or ""),
            constraints=tuple(
                ConstraintSpec(**dict(value))
                for value in (raw.get("constraints") or ())
            ),
            frozen_component_fingerprints=dict(
                raw.get("frozen_component_fingerprints") or {}
            ),
            sampler_id=str(raw.get("sampler_id") or "tpe"),
            seed=int(raw.get("seed", 42)),
            enabled=bool(raw.get("enabled", False)),
            schema_version=int(raw.get("schema_version", PHASE8_SCHEMA_VERSION)),
        )


def load_phase8_experiment_spec(path: str | Path) -> Phase8ExperimentSpec:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Phase 8 YAML must contain a mapping")
    return Phase8ExperimentSpec.from_mapping(payload)


__all__ = [
    "LAYER_COMPONENT_TYPES",
    "PHASE8_SCHEMA_VERSION",
    "PHASE8_SUPPORTED_SAMPLERS",
    "OptimizationLayer",
    "Phase8ExperimentSpec",
    "Phase8FoldConfig",
    "Phase8ObjectiveSpec",
    "load_phase8_experiment_spec",
]
