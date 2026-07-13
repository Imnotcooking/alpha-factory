"""Deterministic, side-effect-free orchestration contracts for Paper 1.

This module sequences injected components and records explicit scaffold skips.
It neither reads configuration nor writes artifacts.  Scientific component
implementations enter only at their gated stages; the Stage 2 smoke runner can
therefore prove plumbing without manufacturing HMM, VQ-VAE, or economic output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


STAGE_OWNER = 2


class PipelineMode(str, Enum):
    SYNTHETIC_SMOKE = "synthetic_smoke"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class PipelineStage(str, Enum):
    CONFIG = "config"
    SYNTHETIC_INPUT = "synthetic_input"
    CONTINUOUS_SERIES = "continuous_series"
    FEATURES = "features"
    FOLDS = "folds"
    TARGETS = "targets"
    PREPROCESSING = "preprocessing"
    BASELINES = "baselines"
    HMM_MODELS = "hmm_models"
    FILTERING = "filtering"
    ALIGNMENT = "alignment"
    VQVAE = "vqvae"
    DIAGNOSTICS = "diagnostics"
    EVALUATION = "evaluation"
    RISK_THROTTLE = "risk_throttle"
    REPORTING = "reporting"
    MANIFEST = "manifest"


PIPELINE_STAGE_ORDER = tuple(PipelineStage)


class StageStatus(str, Enum):
    SUCCEEDED = "succeeded"
    SKIPPED_SCAFFOLD = "skipped_scaffold"


class PipelinePermission(str, Enum):
    """Privileged capabilities a component must declare before execution."""

    MARKET_DATA = "market_data"
    NETWORK = "network"
    HOLDOUT = "holdout"
    MANUSCRIPT_WRITES = "manuscript_writes"


class PipelineExecutionError(RuntimeError):
    """Wrap a component failure with stable stage and component identifiers."""

    def __init__(self, *, stage: PipelineStage, component_id: str, message: str):
        super().__init__(f"{stage.value}/{component_id}: {message}")
        self.stage = stage
        self.component_id = component_id


@dataclass(frozen=True)
class PipelineGuards:
    """Capabilities the caller explicitly authorizes for a run."""

    market_data_allowed: bool = False
    network_allowed: bool = False
    holdout_allowed: bool = False
    manuscript_writes_allowed: bool = False

    def __post_init__(self) -> None:
        values = (
            self.market_data_allowed,
            self.network_allowed,
            self.holdout_allowed,
            self.manuscript_writes_allowed,
        )
        if any(not isinstance(value, bool) for value in values):
            raise TypeError("Pipeline guard values must be booleans.")

    def allows(self, permission: PipelinePermission) -> bool:
        """Return whether one explicitly declared permission is authorized."""

        if not isinstance(permission, PipelinePermission):
            raise TypeError("permission must be a PipelinePermission value.")
        return {
            PipelinePermission.MARKET_DATA: self.market_data_allowed,
            PipelinePermission.NETWORK: self.network_allowed,
            PipelinePermission.HOLDOUT: self.holdout_allowed,
            PipelinePermission.MANUSCRIPT_WRITES: self.manuscript_writes_allowed,
        }[permission]


@dataclass(frozen=True)
class PipelineRequest:
    """Resolved, non-scientific execution envelope."""

    run_id: str
    mode: PipelineMode
    seed: int
    artifact_root: Path
    guards: PipelineGuards = field(default_factory=PipelineGuards)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty.")
        if not isinstance(self.mode, PipelineMode):
            raise TypeError("mode must be a PipelineMode value.")
        if not isinstance(self.artifact_root, Path):
            raise TypeError("artifact_root must be a pathlib.Path.")
        if not isinstance(self.guards, PipelineGuards):
            raise TypeError("guards must be a PipelineGuards value.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        if self.mode is PipelineMode.SYNTHETIC_SMOKE:
            enabled = (
                self.guards.market_data_allowed,
                self.guards.network_allowed,
                self.guards.holdout_allowed,
                self.guards.manuscript_writes_allowed,
            )
            if any(enabled):
                raise ValueError(
                    "Synthetic smoke runs must forbid market data, network, holdout, "
                    "and manuscript writes."
                )
        if self.mode is not PipelineMode.HOLDOUT and self.guards.holdout_allowed:
            raise ValueError("Holdout authorization is valid only in holdout mode.")
        if self.mode is PipelineMode.HOLDOUT and not self.guards.holdout_allowed:
            raise ValueError("Holdout mode requires explicit holdout authorization.")


@dataclass(frozen=True)
class ArtifactReference:
    """An intended run-relative artifact, without performing filesystem I/O."""

    kind: str
    relative_path: str
    scientific_evidence: bool

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Artifact kind must be non-empty.")
        if not isinstance(self.scientific_evidence, bool):
            raise TypeError("scientific_evidence must be a boolean.")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Artifact paths must be normalized run-relative paths.")


@dataclass(frozen=True)
class StageOutput:
    """Immutable contribution from one injected pipeline component."""

    status: StageStatus
    updates: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[ArtifactReference, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageStatus):
            raise TypeError("status must be a StageStatus value.")
        object.__setattr__(self, "updates", MappingProxyType(dict(self.updates)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.status is StageStatus.SKIPPED_SCAFFOLD and not self.note.strip():
            raise ValueError("Scaffold skips require an explanatory note.")

    @classmethod
    def skipped_scaffold(cls, reason: str) -> StageOutput:
        """Describe an intentionally unavailable later-stage capability."""

        return cls(status=StageStatus.SKIPPED_SCAFFOLD, note=reason)


@dataclass(frozen=True)
class StageRecord:
    """Deterministic stage ledger record; wall-clock timing is excluded."""

    stage: PipelineStage
    component_id: str
    owner_stage: int
    status: StageStatus
    diagnostic_keys: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class PipelineContext:
    """Read-only values visible to the next component in the sequence."""

    request: PipelineRequest
    values: Mapping[str, Any] = field(default_factory=dict)
    records: tuple[StageRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def advance(
        self,
        *,
        output: StageOutput,
        record: StageRecord,
    ) -> PipelineContext:
        overlap = set(self.values).intersection(output.updates)
        if overlap:
            raise ValueError(
                "Pipeline components cannot silently overwrite context keys: "
                f"{sorted(overlap)}"
            )
        values = dict(self.values)
        values.update(output.updates)
        return PipelineContext(
            request=self.request,
            values=values,
            records=self.records + (record,),
        )


@runtime_checkable
class PipelineComponent(Protocol):
    """One dependency-injected, side-effect-controlled pipeline stage."""

    @property
    def component_id(self) -> str:
        """Stable identifier included in the stage ledger."""

    @property
    def stage(self) -> PipelineStage:
        """Unique stage occupied by this component."""

    @property
    def owner_stage(self) -> int:
        """Research-plan stage that owns the scientific implementation."""

    @property
    def required_permissions(self) -> frozenset[PipelinePermission]:
        """Privileged capabilities needed before ``run`` may be called."""

    def run(self, context: PipelineContext) -> StageOutput:
        """Return values and artifact references without mutating context."""


@dataclass(frozen=True)
class PipelineResult:
    """Final in-memory output of a successfully orchestrated component list."""

    request: PipelineRequest
    values: Mapping[str, Any]
    records: tuple[StageRecord, ...]
    artifacts: tuple[ArtifactReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        if self.request.mode is PipelineMode.SYNTHETIC_SMOKE and any(
            artifact.scientific_evidence for artifact in self.artifacts
        ):
            raise ValueError(
                "A synthetic PipelineResult cannot contain scientific-evidence artifacts."
            )
        if not self.request.guards.manuscript_writes_allowed and any(
            record.stage is PipelineStage.REPORTING
            and record.status is StageStatus.SUCCEEDED
            for record in self.records
        ):
            raise ValueError(
                "A successful reporting stage requires manuscript-write authorization."
            )

    @property
    def scaffolded_stages(self) -> tuple[PipelineStage, ...]:
        return tuple(
            record.stage
            for record in self.records
            if record.status is StageStatus.SKIPPED_SCAFFOLD
        )


def execute_pipeline(
    request: PipelineRequest,
    components: Sequence[PipelineComponent],
) -> PipelineResult:
    """Run injected components once each, in canonical order, without I/O.

    Components own any later side effects through explicit adapters.  This
    function intentionally omits timestamps and elapsed durations so its stage
    ledger remains suitable for deterministic smoke artifacts.
    """

    context = PipelineContext(request=request)
    artifacts: list[ArtifactReference] = []
    seen_stages: set[PipelineStage] = set()
    prior_order = -1
    order = {stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)}

    for component in components:
        stage = component.stage
        if stage in seen_stages:
            raise ValueError(f"Pipeline stage {stage.value!r} was supplied more than once.")
        stage_order = order[stage]
        if stage_order <= prior_order:
            raise ValueError("Pipeline components must follow PIPELINE_STAGE_ORDER.")
        if component.owner_stage < STAGE_OWNER:
            raise ValueError("Component owner_stage cannot precede the package scaffold.")

        permissions = _component_permissions(component)
        unauthorized = tuple(
            permission
            for permission in permissions
            if not request.guards.allows(permission)
        )
        if (
            stage is PipelineStage.REPORTING
            and not request.guards.manuscript_writes_allowed
        ):
            unauthorized = tuple(
                dict.fromkeys((*unauthorized, PipelinePermission.MANUSCRIPT_WRITES))
            )
        if unauthorized:
            names = ", ".join(permission.value for permission in unauthorized)
            raise PipelineExecutionError(
                stage=stage,
                component_id=component.component_id,
                message=f"run guards do not authorize: {names}",
            )

        try:
            output = component.run(context)
        except Exception as exc:
            raise PipelineExecutionError(
                stage=stage,
                component_id=component.component_id,
                message=str(exc),
            ) from exc

        if (
            output.status is StageStatus.SKIPPED_SCAFFOLD
            and component.owner_stage <= STAGE_OWNER
        ):
            raise ValueError("Stage 2 components cannot be marked as later-stage scaffolds.")

        _validate_artifact_authorization(
            request=request,
            stage=stage,
            component_id=component.component_id,
            artifacts=output.artifacts,
        )

        artifact_paths = tuple(artifact.relative_path for artifact in output.artifacts)
        existing_paths = {artifact.relative_path for artifact in artifacts}
        duplicates = existing_paths.intersection(artifact_paths)
        if len(artifact_paths) != len(set(artifact_paths)):
            duplicates.update(
                path for path in artifact_paths if artifact_paths.count(path) > 1
            )
        if duplicates:
            raise ValueError(f"Artifact paths must be unique: {sorted(duplicates)}")

        record = StageRecord(
            stage=stage,
            component_id=component.component_id,
            owner_stage=component.owner_stage,
            status=output.status,
            diagnostic_keys=tuple(sorted(output.diagnostics)),
            artifact_paths=artifact_paths,
            note=output.note,
        )
        context = context.advance(output=output, record=record)
        artifacts.extend(output.artifacts)
        seen_stages.add(stage)
        prior_order = stage_order

    return PipelineResult(
        request=request,
        values=context.values,
        records=context.records,
        artifacts=tuple(artifacts),
    )


def _component_permissions(
    component: PipelineComponent,
) -> frozenset[PipelinePermission]:
    """Read declared permissions; legacy components are treated as unprivileged."""

    raw = getattr(component, "required_permissions", frozenset())
    try:
        permissions = frozenset(raw)
    except TypeError as exc:
        raise TypeError("required_permissions must be an iterable of permissions.") from exc
    invalid = [item for item in permissions if not isinstance(item, PipelinePermission)]
    if invalid:
        raise TypeError("required_permissions must contain PipelinePermission values.")
    return permissions


def _validate_artifact_authorization(
    *,
    request: PipelineRequest,
    stage: PipelineStage,
    component_id: str,
    artifacts: tuple[ArtifactReference, ...],
) -> None:
    if request.mode is PipelineMode.SYNTHETIC_SMOKE:
        evidentiary = [
            artifact.relative_path for artifact in artifacts if artifact.scientific_evidence
        ]
        if evidentiary:
            raise PipelineExecutionError(
                stage=stage,
                component_id=component_id,
                message=(
                    "synthetic runs cannot emit scientific-evidence artifacts: "
                    f"{sorted(evidentiary)}"
                ),
            )

    for artifact in artifacts:
        path = PurePosixPath(artifact.relative_path)
        root = path.parts[0].lower()
        kind = artifact.kind.lower()
        if not request.guards.holdout_allowed and (
            root == "holdout" or kind.startswith("holdout")
        ):
            raise PipelineExecutionError(
                stage=stage,
                component_id=component_id,
                message=f"holdout artifact is not authorized: {artifact.relative_path}",
            )
        manuscript_artifact = root in {"paper", "manuscript", "tables", "figures"} or (
            kind in {"manuscript", "paper_table", "paper_figure"}
        )
        if manuscript_artifact and not request.guards.manuscript_writes_allowed:
            raise PipelineExecutionError(
                stage=stage,
                component_id=component_id,
                message=f"manuscript artifact is not authorized: {artifact.relative_path}",
            )


__all__ = [
    "ArtifactReference",
    "PIPELINE_STAGE_ORDER",
    "PipelineComponent",
    "PipelineContext",
    "PipelineExecutionError",
    "PipelineGuards",
    "PipelineMode",
    "PipelinePermission",
    "PipelineRequest",
    "PipelineResult",
    "PipelineStage",
    "STAGE_OWNER",
    "StageOutput",
    "StageRecord",
    "StageStatus",
    "execute_pipeline",
]
