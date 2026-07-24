"""Versioned, predeclared success criteria for research experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUCCESS_CRITERIA_PATH = (
    REPO_ROOT / "config" / "research" / "success_criteria.yaml"
)
SUCCESS_CRITERIA_SCHEMA_VERSION = 1
VALID_RESEARCH_OBJECTS = {"factor", "sleeve", "strategy", "router", "model"}
VALID_DIRECTIONS = {"maximize", "minimize"}
VALID_OPERATORS = {"<=", ">=", "<", ">", "=="}


class CriterionDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class MetricGate:
    name: str
    metric: str
    operator: str
    threshold: float
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.metric).strip():
            raise ValueError("gate name and metric cannot be empty")
        if self.operator not in VALID_OPERATORS:
            raise ValueError(
                f"gate operator must be one of {sorted(VALID_OPERATORS)}"
            )
        if not math.isfinite(float(self.threshold)):
            raise ValueError("gate threshold must be finite")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MetricGate":
        return cls(
            name=str(payload.get("name") or payload.get("metric") or ""),
            metric=str(payload.get("metric") or ""),
            operator=str(payload.get("operator") or ""),
            threshold=float(payload.get("threshold")),
            description=str(payload.get("description") or ""),
        )


@dataclass(frozen=True, slots=True)
class SuccessCriterionSpec:
    profile_id: str
    research_object: str
    decision_sample: str
    economic_question: str
    primary_metric: str
    direction: str
    comparator_metric: str | None = None
    minimum_improvement: float = 0.0
    absolute_floor: float | None = None
    gates: tuple[MetricGate, ...] = ()
    required_metrics: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    status: str = "active"
    schema_version: int = SUCCESS_CRITERIA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        research_object = str(self.research_object).strip().lower()
        direction = str(self.direction).strip().lower()
        primary_metric = str(self.primary_metric).strip()
        if not profile_id or not primary_metric:
            raise ValueError("profile_id and primary_metric cannot be empty")
        if research_object not in VALID_RESEARCH_OBJECTS:
            raise ValueError(
                f"research_object must be one of {sorted(VALID_RESEARCH_OBJECTS)}"
            )
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(VALID_DIRECTIONS)}"
            )
        if int(self.schema_version) != SUCCESS_CRITERIA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported success-criterion schema {self.schema_version}"
            )
        comparator_metric = _optional_text(self.comparator_metric)
        minimum_improvement = float(self.minimum_improvement)
        if minimum_improvement < 0 or not math.isfinite(minimum_improvement):
            raise ValueError("minimum_improvement must be finite and non-negative")
        absolute_floor = (
            None if self.absolute_floor is None else float(self.absolute_floor)
        )
        if absolute_floor is not None and not math.isfinite(absolute_floor):
            raise ValueError("absolute_floor must be finite or null")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "research_object", research_object)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "primary_metric", primary_metric)
        object.__setattr__(self, "comparator_metric", comparator_metric)
        object.__setattr__(self, "minimum_improvement", minimum_improvement)
        object.__setattr__(self, "absolute_floor", absolute_floor)
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(
            self,
            "required_metrics",
            tuple(str(item).strip() for item in self.required_metrics if str(item).strip()),
        )
        object.__setattr__(
            self,
            "required_artifacts",
            tuple(str(item).strip() for item in self.required_artifacts if str(item).strip()),
        )

    @classmethod
    def from_mapping(
        cls,
        profile_id: str,
        payload: Mapping[str, Any],
    ) -> "SuccessCriterionSpec":
        primary = payload.get("primary") or {}
        if not isinstance(primary, Mapping):
            raise ValueError(f"{profile_id} primary must be a mapping")
        raw_gates = payload.get("gates") or ()
        if not isinstance(raw_gates, (list, tuple)):
            raise ValueError(f"{profile_id} gates must be a list")
        return cls(
            profile_id=profile_id,
            research_object=str(payload.get("research_object") or ""),
            decision_sample=str(payload.get("decision_sample") or ""),
            economic_question=str(payload.get("economic_question") or ""),
            primary_metric=str(primary.get("metric") or ""),
            direction=str(primary.get("direction") or "maximize"),
            comparator_metric=_optional_text(primary.get("comparator_metric")),
            minimum_improvement=float(primary.get("minimum_improvement", 0.0)),
            absolute_floor=_optional_float(primary.get("absolute_floor")),
            gates=tuple(MetricGate.from_mapping(item) for item in raw_gates),
            required_metrics=tuple(payload.get("required_metrics") or ()),
            required_artifacts=tuple(payload.get("required_artifacts") or ()),
            status=str(payload.get("status") or "active"),
            schema_version=int(
                payload.get("schema_version", SUCCESS_CRITERIA_SCHEMA_VERSION)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "research_object": self.research_object,
            "decision_sample": self.decision_sample,
            "economic_question": self.economic_question,
            "status": self.status,
            "primary": {
                "metric": self.primary_metric,
                "direction": self.direction,
                "comparator_metric": self.comparator_metric,
                "minimum_improvement": self.minimum_improvement,
                "absolute_floor": self.absolute_floor,
            },
            "gates": [asdict(gate) for gate in self.gates],
            "required_metrics": list(self.required_metrics),
            "required_artifacts": list(self.required_artifacts),
        }

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    name: str
    metric: str
    value: float
    operator: str
    threshold: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SuccessCriterionResult:
    profile_id: str
    profile_fingerprint: str
    decision: CriterionDecision
    primary_value: float | None
    comparator_value: float | None
    improvement: float | None
    primary_floor_passed: bool | None
    comparator_passed: bool | None
    gates: tuple[GateEvaluation, ...]
    missing_metrics: tuple[str, ...]
    failed_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.decision == CriterionDecision.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_fingerprint": self.profile_fingerprint,
            "decision": self.decision.value,
            "passed": self.passed,
            "primary_value": self.primary_value,
            "comparator_value": self.comparator_value,
            "improvement": self.improvement,
            "primary_floor_passed": self.primary_floor_passed,
            "comparator_passed": self.comparator_passed,
            "gates": [gate.to_dict() for gate in self.gates],
            "missing_metrics": list(self.missing_metrics),
            "failed_reasons": list(self.failed_reasons),
        }


class SuccessCriterionRegistry:
    def __init__(
        self,
        profiles: Mapping[str, SuccessCriterionSpec],
        *,
        registry_id: str,
        as_of: str = "",
    ) -> None:
        self.profiles = dict(profiles)
        self.registry_id = str(registry_id).strip()
        self.as_of = str(as_of).strip()
        if not self.registry_id or not self.profiles:
            raise ValueError("success-criterion registry requires ID and profiles")

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_SUCCESS_CRITERIA_PATH,
    ) -> "SuccessCriterionRegistry":
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"success-criterion registry not found: {source}")
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("success-criterion registry must contain a mapping")
        if int(payload.get("schema_version", 0)) != SUCCESS_CRITERIA_SCHEMA_VERSION:
            raise ValueError("unsupported success-criterion registry schema")
        raw_profiles = payload.get("profiles") or {}
        if not isinstance(raw_profiles, Mapping) or not raw_profiles:
            raise ValueError("success-criterion registry requires profiles")
        profiles = {
            str(profile_id): SuccessCriterionSpec.from_mapping(
                str(profile_id), profile
            )
            for profile_id, profile in raw_profiles.items()
        }
        return cls(
            profiles,
            registry_id=str(payload.get("registry_id") or ""),
            as_of=str(payload.get("as_of") or ""),
        )

    def resolve(self, profile_id: str) -> SuccessCriterionSpec:
        profile_id = str(profile_id).strip()
        if profile_id not in self.profiles:
            raise KeyError(
                f"Unknown success-criterion profile {profile_id!r}; expected one of "
                f"{sorted(self.profiles)}"
            )
        return self.profiles[profile_id]


def evaluate_success_criterion(
    spec: SuccessCriterionSpec,
    metrics: Mapping[str, Any],
) -> SuccessCriterionResult:
    numeric = _finite_metrics(metrics)
    needed = {
        spec.primary_metric,
        *spec.required_metrics,
        *(gate.metric for gate in spec.gates),
    }
    if spec.comparator_metric:
        needed.add(spec.comparator_metric)
    missing = tuple(sorted(metric for metric in needed if metric not in numeric))
    if missing:
        return SuccessCriterionResult(
            profile_id=spec.profile_id,
            profile_fingerprint=spec.fingerprint,
            decision=CriterionDecision.INCOMPLETE,
            primary_value=numeric.get(spec.primary_metric),
            comparator_value=(
                numeric.get(spec.comparator_metric)
                if spec.comparator_metric
                else None
            ),
            improvement=None,
            primary_floor_passed=None,
            comparator_passed=None,
            gates=(),
            missing_metrics=missing,
            failed_reasons=("required metrics are missing or non-finite",),
        )

    primary_value = numeric[spec.primary_metric]
    floor_passed = None
    failed_reasons: list[str] = []
    if spec.absolute_floor is not None:
        floor_passed = _directional_floor_passed(
            primary_value, spec.absolute_floor, spec.direction
        )
        if not floor_passed:
            failed_reasons.append("primary metric missed its absolute floor")

    comparator_value = None
    improvement = None
    comparator_passed = None
    if spec.comparator_metric:
        comparator_value = numeric[spec.comparator_metric]
        improvement = (
            primary_value - comparator_value
            if spec.direction == "maximize"
            else comparator_value - primary_value
        )
        comparator_passed = improvement >= spec.minimum_improvement
        if not comparator_passed:
            failed_reasons.append("primary metric did not beat the frozen comparator")

    gate_results: list[GateEvaluation] = []
    for gate in spec.gates:
        value = numeric[gate.metric]
        passed = _compare(value, gate.operator, gate.threshold)
        gate_results.append(
            GateEvaluation(
                name=gate.name,
                metric=gate.metric,
                value=value,
                operator=gate.operator,
                threshold=float(gate.threshold),
                passed=passed,
            )
        )
        if not passed:
            failed_reasons.append(f"gate failed: {gate.name}")

    decision = (
        CriterionDecision.FAIL if failed_reasons else CriterionDecision.PASS
    )
    return SuccessCriterionResult(
        profile_id=spec.profile_id,
        profile_fingerprint=spec.fingerprint,
        decision=decision,
        primary_value=primary_value,
        comparator_value=comparator_value,
        improvement=improvement,
        primary_floor_passed=floor_passed,
        comparator_passed=comparator_passed,
        gates=tuple(gate_results),
        missing_metrics=(),
        failed_reasons=tuple(failed_reasons),
    )


def attach_success_criterion_attrs(
    frame: pd.DataFrame,
    criterion: SuccessCriterionSpec,
) -> pd.DataFrame:
    frame.attrs["success_criterion_profile_id"] = criterion.profile_id
    frame.attrs["success_criterion_fingerprint"] = criterion.fingerprint
    frame.attrs["success_criterion"] = criterion.to_dict()
    frame.attrs["success_criterion_status"] = "declared_not_evaluated"
    return frame


def attach_success_criterion_result_attrs(
    frame: pd.DataFrame,
    result: SuccessCriterionResult,
) -> pd.DataFrame:
    expected = str(frame.attrs.get("success_criterion_fingerprint") or "")
    if expected and expected != result.profile_fingerprint:
        raise ValueError("criterion result does not match the frame's frozen profile")
    frame.attrs["success_criterion_result"] = result.to_dict()
    frame.attrs["success_criterion_status"] = result.decision.value
    return frame


def success_criterion_manifest_payload(frame: pd.DataFrame) -> dict[str, Any]:
    criterion = frame.attrs.get("success_criterion") or {}
    result = frame.attrs.get("success_criterion_result") or {}
    return {
        "status": str(
            frame.attrs.get("success_criterion_status") or "not_declared"
        ),
        "profile_id": frame.attrs.get("success_criterion_profile_id"),
        "profile_fingerprint": frame.attrs.get(
            "success_criterion_fingerprint"
        ),
        "definition": dict(criterion) if isinstance(criterion, Mapping) else {},
        "evaluation": dict(result) if isinstance(result, Mapping) else {},
    }


def _finite_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, value in metrics.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            output[str(name)] = numeric
    return output


def _directional_floor_passed(
    value: float,
    floor: float,
    direction: str,
) -> bool:
    return value >= floor if direction == "maximize" else value <= floor


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    return math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


__all__ = [
    "CriterionDecision",
    "DEFAULT_SUCCESS_CRITERIA_PATH",
    "GateEvaluation",
    "MetricGate",
    "SUCCESS_CRITERIA_SCHEMA_VERSION",
    "SuccessCriterionRegistry",
    "SuccessCriterionResult",
    "SuccessCriterionSpec",
    "attach_success_criterion_attrs",
    "attach_success_criterion_result_attrs",
    "evaluate_success_criterion",
    "success_criterion_manifest_payload",
]
