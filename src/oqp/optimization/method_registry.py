"""Purpose-to-method registry for governed optimization work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METHOD_REGISTRY = REPO_ROOT / "config/research/optimization_methods.yaml"


@dataclass(frozen=True, slots=True)
class OptimizationMethodProfile:
    method_id: str
    label: str
    family: str
    engine: str
    status: str
    good_for: str
    warning: str

    @classmethod
    def from_mapping(
        cls, method_id: str, raw: dict[str, Any]
    ) -> "OptimizationMethodProfile":
        required = ("label", "family", "engine", "status", "good_for", "warning")
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"optimization method {method_id} is missing: {', '.join(missing)}"
            )
        return cls(
            method_id=str(method_id).strip(),
            label=str(raw["label"]).strip(),
            family=str(raw["family"]).strip(),
            engine=str(raw["engine"]).strip(),
            status=str(raw["status"]).strip(),
            good_for=str(raw["good_for"]).strip(),
            warning=str(raw["warning"]).strip(),
        )


@dataclass(frozen=True, slots=True)
class OptimizationPurposeProfile:
    purpose_id: str
    label: str
    layer: str
    job_type: str
    status: str
    objective_profile_id: str | None
    primary_method: str
    benchmark_method: str | None
    alternative_methods: tuple[str, ...]
    purpose: str
    rationale: str
    blocking_reason: str = ""

    @classmethod
    def from_mapping(
        cls, purpose_id: str, raw: dict[str, Any]
    ) -> "OptimizationPurposeProfile":
        required = (
            "label",
            "layer",
            "job_type",
            "status",
            "primary_method",
            "purpose",
            "rationale",
        )
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"optimization purpose {purpose_id} is missing: {', '.join(missing)}"
            )
        objective_profile = raw.get("objective_profile_id")
        benchmark = raw.get("benchmark_method")
        return cls(
            purpose_id=str(purpose_id).strip(),
            label=str(raw["label"]).strip(),
            layer=str(raw["layer"]).strip(),
            job_type=str(raw["job_type"]).strip(),
            status=str(raw["status"]).strip(),
            objective_profile_id=(
                str(objective_profile).strip() if objective_profile else None
            ),
            primary_method=str(raw["primary_method"]).strip(),
            benchmark_method=str(benchmark).strip() if benchmark else None,
            alternative_methods=tuple(
                str(item).strip()
                for item in raw.get("alternative_methods", ())
                if str(item).strip()
            ),
            purpose=str(raw["purpose"]).strip(),
            rationale=str(raw["rationale"]).strip(),
            blocking_reason=str(raw.get("blocking_reason", "")).strip(),
        )

    @property
    def method_ids(self) -> tuple[str, ...]:
        ordered = [self.primary_method]
        if self.benchmark_method:
            ordered.append(self.benchmark_method)
        ordered.extend(self.alternative_methods)
        return tuple(dict.fromkeys(ordered))


class OptimizationMethodRegistry:
    def __init__(
        self,
        methods: dict[str, OptimizationMethodProfile],
        purposes: dict[str, OptimizationPurposeProfile],
        *,
        registry_id: str,
        schema_version: int,
        source_path: Path,
    ) -> None:
        self.methods = dict(methods)
        self.purposes = dict(purposes)
        self.registry_id = str(registry_id)
        self.schema_version = int(schema_version)
        self.source_path = source_path
        self._validate()

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_METHOD_REGISTRY
    ) -> "OptimizationMethodRegistry":
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        raw_methods = payload.get("methods") or {}
        raw_purposes = payload.get("purposes") or {}
        if not isinstance(raw_methods, dict) or not isinstance(raw_purposes, dict):
            raise ValueError("optimization methods and purposes must be mappings")
        methods = {
            str(method_id): OptimizationMethodProfile.from_mapping(
                str(method_id), dict(raw)
            )
            for method_id, raw in raw_methods.items()
        }
        purposes = {
            str(purpose_id): OptimizationPurposeProfile.from_mapping(
                str(purpose_id), dict(raw)
            )
            for purpose_id, raw in raw_purposes.items()
        }
        if not methods or not purposes:
            raise ValueError("optimization method registry cannot be empty")
        return cls(
            methods,
            purposes,
            registry_id=str(payload.get("registry_id") or ""),
            schema_version=int(payload.get("schema_version", 1)),
            source_path=source,
        )

    def _validate(self) -> None:
        for purpose in self.purposes.values():
            unknown = [
                method_id
                for method_id in purpose.method_ids
                if method_id not in self.methods
            ]
            if unknown:
                raise ValueError(
                    f"optimization purpose {purpose.purpose_id} references "
                    f"unknown methods: {', '.join(unknown)}"
                )
            if purpose.status == "blocked" and not purpose.blocking_reason:
                raise ValueError(
                    f"blocked optimization purpose {purpose.purpose_id} "
                    "requires a blocking_reason"
                )

    def resolve_purpose(self, purpose_id: str) -> OptimizationPurposeProfile:
        key = str(purpose_id).strip()
        try:
            return self.purposes[key]
        except KeyError as exc:
            raise KeyError(f"unknown optimization purpose: {key}") from exc

    def resolve_method(self, method_id: str) -> OptimizationMethodProfile:
        key = str(method_id).strip()
        try:
            return self.methods[key]
        except KeyError as exc:
            raise KeyError(f"unknown optimization method: {key}") from exc

    def purpose_inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "purpose_id": purpose.purpose_id,
                "label": purpose.label,
                "layer": purpose.layer,
                "job_type": purpose.job_type,
                "status": purpose.status,
                "primary_method": self.methods[purpose.primary_method].label,
                "benchmark_method": (
                    self.methods[purpose.benchmark_method].label
                    if purpose.benchmark_method
                    else "Not applicable"
                ),
                "objective_profile_id": purpose.objective_profile_id or "",
            }
            for purpose in self.purposes.values()
        ]


__all__ = [
    "DEFAULT_METHOD_REGISTRY",
    "OptimizationMethodProfile",
    "OptimizationMethodRegistry",
    "OptimizationPurposeProfile",
]
