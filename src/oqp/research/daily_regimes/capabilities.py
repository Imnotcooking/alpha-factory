"""Explicit capability and paper-eligibility declarations.

An implementation existing in the repository is not sufficient evidence for a
paper claim.  The registry keeps implementation maturity and paper eligibility
separate so synthetic or exploratory code cannot be promoted accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class CapabilityStage(str, Enum):
    DECLARED = "declared"
    IMPLEMENTED = "implemented"
    SYNTHETIC_VERIFIED = "synthetic_verified"
    VALIDATION_VERIFIED = "validation_verified"
    HOLDOUT_FROZEN = "holdout_frozen"
    PAPER_READY = "paper_ready"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class ResearchCapability:
    capability_id: str
    module: str
    stage: CapabilityStage
    paper_eligible: bool
    eligibility_reason: str
    required_artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.capability_id.strip():
            raise ValueError("capability_id cannot be empty.")
        if not self.module.strip():
            raise ValueError("Capability module cannot be empty.")
        if not self.eligibility_reason.strip():
            raise ValueError("eligibility_reason must be explicit.")
        if self.paper_eligible and self.stage is not CapabilityStage.PAPER_READY:
            raise ValueError(
                "A capability can be paper eligible only at the paper_ready stage."
            )
        if len(set(self.required_artifacts)) != len(self.required_artifacts):
            raise ValueError("required_artifacts cannot contain duplicates.")

    def advance(
        self,
        stage: CapabilityStage,
        *,
        paper_eligible: bool = False,
        reason: str,
        required_artifacts: tuple[str, ...] | None = None,
    ) -> "ResearchCapability":
        return replace(
            self,
            stage=stage,
            paper_eligible=paper_eligible,
            eligibility_reason=reason,
            required_artifacts=(
                self.required_artifacts
                if required_artifacts is None
                else tuple(required_artifacts)
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "capability_id": self.capability_id,
            "module": self.module,
            "stage": self.stage.value,
            "paper_eligible": self.paper_eligible,
            "eligibility_reason": self.eligibility_reason,
            "required_artifacts": list(self.required_artifacts),
        }


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    capabilities: tuple[ResearchCapability, ...]

    def __post_init__(self) -> None:
        identifiers = [item.capability_id for item in self.capabilities]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Capability IDs must be unique.")

    def get(self, capability_id: str) -> ResearchCapability | None:
        return next(
            (
                capability
                for capability in self.capabilities
                if capability.capability_id == capability_id
            ),
            None,
        )

    def require(self, capability_id: str) -> ResearchCapability:
        capability = self.get(capability_id)
        if capability is None:
            raise KeyError(f"Unknown daily-regime capability: {capability_id}")
        return capability

    def with_capability(self, capability: ResearchCapability) -> "CapabilityRegistry":
        retained = tuple(
            item
            for item in self.capabilities
            if item.capability_id != capability.capability_id
        )
        return CapabilityRegistry((*retained, capability))

    def paper_eligible_capabilities(self) -> tuple[ResearchCapability, ...]:
        return tuple(item for item in self.capabilities if item.paper_eligible)

    def is_paper_eligible(self, capability_id: str) -> bool:
        return self.require(capability_id).paper_eligible

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            item.capability_id: item.to_dict()
            for item in sorted(self.capabilities, key=lambda value: value.capability_id)
        }


_SYNTHETIC_VERIFIED = (
    "configuration",
    "contracts",
    "capability_registry",
    "deterministic_seeding",
    "runtime_paths",
    "artifact_manifests",
    "synthetic_fixture",
    "pipeline_orchestration",
    "smoke_runner",
    "continuous_series",
    "features",
    "preprocessing",
)

_DECLARED = (
    "targets",
    "folds",
    "baselines",
    "hmm_models",
    "forward_filtering",
    "state_alignment",
    "vqvae",
    "diagnostics",
    "evaluation",
    "risk_throttle",
    "reporting",
)


def default_capability_registry() -> CapabilityRegistry:
    """Return the conservative Stage 2 capability registry."""

    synthetic_verified = tuple(
        ResearchCapability(
            capability_id=name,
            module=f"oqp.research.daily_regimes.{_module_name(name)}",
            stage=CapabilityStage.SYNTHETIC_VERIFIED,
            paper_eligible=False,
            eligibility_reason=(
                "Verified by the deterministic two-attempt cumulative synthetic gate; "
                "synthetic verification is not empirical evidence."
            ),
        )
        for name in _SYNTHETIC_VERIFIED
    )
    declared = tuple(
        ResearchCapability(
            capability_id=name,
            module=f"oqp.research.daily_regimes.{_module_name(name)}",
            stage=CapabilityStage.DECLARED,
            paper_eligible=False,
            eligibility_reason="Declared in the research plan; implementation evidence is pending.",
        )
        for name in _DECLARED
    )
    return CapabilityRegistry((*synthetic_verified, *declared))


def _module_name(capability_id: str) -> str:
    aliases = {
        "configuration": "config",
        "capability_registry": "capabilities",
        "deterministic_seeding": "seeding",
        "artifact_manifests": "artifacts",
        "synthetic_fixture": "synthetic",
        "pipeline_orchestration": "pipeline",
        "smoke_runner": "smoke",
        "hmm_models": "hmm",
        "forward_filtering": "filtering",
    }
    return aliases.get(capability_id, capability_id)


__all__ = [
    "CapabilityRegistry",
    "CapabilityStage",
    "ResearchCapability",
    "default_capability_registry",
]
