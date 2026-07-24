"""Loader for the Phase 9 objective-profile registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from oqp.research.optimization_objectives.contracts import (
    OptimizationObjectiveProfile,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OBJECTIVE_REGISTRY = REPO_ROOT / "config/research/optimization_objectives.yaml"


class OptimizationObjectiveRegistry:
    def __init__(
        self,
        profiles: dict[str, OptimizationObjectiveProfile],
        *,
        registry_id: str,
        schema_version: int,
        source_path: Path,
    ) -> None:
        self.profiles = dict(profiles)
        self.registry_id = str(registry_id)
        self.schema_version = int(schema_version)
        self.source_path = source_path

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_OBJECTIVE_REGISTRY
    ) -> "OptimizationObjectiveRegistry":
        source = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        raw_profiles = payload.get("profiles") or {}
        if not isinstance(raw_profiles, dict):
            raise ValueError("optimization objective profiles must be a mapping")
        profiles = {
            str(profile_id): OptimizationObjectiveProfile.from_mapping(
                str(profile_id), value
            )
            for profile_id, value in raw_profiles.items()
        }
        if not profiles:
            raise ValueError("optimization objective registry cannot be empty")
        return cls(
            profiles,
            registry_id=str(payload.get("registry_id") or ""),
            schema_version=int(payload.get("schema_version", 1)),
            source_path=source,
        )

    def resolve(self, profile_id: str) -> OptimizationObjectiveProfile:
        key = str(profile_id).strip()
        try:
            profile = self.profiles[key]
        except KeyError as exc:
            raise KeyError(f"unknown optimization objective profile: {key}") from exc
        if profile.status != "active":
            raise ValueError(f"optimization objective profile {key} is not active")
        return profile

    def inventory(self) -> list[dict[str, Any]]:
        return [
            {
                "profile_id": profile.profile_id,
                "layer": profile.layer,
                "status": profile.status,
                "objective_count": len(profile.objectives),
                "constraint_count": len(profile.constraints),
                "upstream_requirement_count": len(
                    profile.upstream_requirements
                ),
                "fingerprint": profile.fingerprint,
                "economic_question": profile.economic_question,
            }
            for profile in self.profiles.values()
        ]


__all__ = ["DEFAULT_OBJECTIVE_REGISTRY", "OptimizationObjectiveRegistry"]
