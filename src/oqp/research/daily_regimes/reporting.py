"""Evidence-gated reporting contracts for manuscript tables and figures.

Renderers are supplied later.  The contracts prevent synthetic smoke output,
unfrozen artifacts, and placeholders from being marked eligible for empirical
paper claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


STAGE_OWNER = 13


class EvidenceTier(str, Enum):
    SYNTHETIC_SMOKE = "synthetic_smoke"
    FROZEN_PROTOCOL = "frozen_protocol"
    FROZEN_VALIDATION = "frozen_validation"
    FROZEN_HOLDOUT = "frozen_holdout"


class ReportSectionRole(str, Enum):
    METHODS = "methods"
    VALIDATION_APPENDIX = "validation_appendix"
    PRIMARY_EMPIRICAL_RESULTS = "primary_empirical_results"


class ReportArtifactKind(str, Enum):
    LATEX_TABLE = "latex_table"
    VECTOR_FIGURE = "vector_figure"
    MACHINE_READABLE_SUMMARY = "machine_readable_summary"


@dataclass(frozen=True)
class ReportSourceArtifact:
    artifact_id: str
    sha256: str
    manifest_hash: str
    evidence_tier: EvidenceTier
    frozen: bool
    placeholder: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_tier, EvidenceTier):
            raise TypeError("evidence_tier must be an EvidenceTier")
        if not self.artifact_id or not self.sha256 or not self.manifest_hash:
            raise ValueError("report source provenance is required")
        if self.placeholder and self.frozen:
            raise ValueError("placeholder artifacts cannot be frozen evidence")


@dataclass(frozen=True)
class ReportRequest:
    report_id: str
    kind: ReportArtifactKind
    section_role: ReportSectionRole
    destination_name: str
    sources: tuple[ReportSourceArtifact, ...]
    contains_empirical_claims: bool

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ReportArtifactKind):
            raise TypeError("kind must be a ReportArtifactKind")
        if not isinstance(self.section_role, ReportSectionRole):
            raise TypeError("section_role must be a ReportSectionRole")
        if not self.report_id or not self.destination_name or not self.sources:
            raise ValueError("report id, destination, and source artifacts are required")
        source_ids = [source.artifact_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("report source artifact ids must be unique")
        if self.section_role is ReportSectionRole.METHODS and self.contains_empirical_claims:
            raise ValueError("methods artifacts may not be labelled as empirical claims")

    @property
    def paper_eligible(self) -> bool:
        if any(not source.frozen or source.placeholder for source in self.sources):
            return False
        tiers = {source.evidence_tier for source in self.sources}
        if EvidenceTier.SYNTHETIC_SMOKE in tiers:
            return False
        if self.section_role is ReportSectionRole.METHODS:
            return tiers.issubset(
                {
                    EvidenceTier.FROZEN_PROTOCOL,
                    EvidenceTier.FROZEN_VALIDATION,
                    EvidenceTier.FROZEN_HOLDOUT,
                }
            )
        if self.section_role is ReportSectionRole.VALIDATION_APPENDIX:
            return tiers.issubset(
                {EvidenceTier.FROZEN_VALIDATION, EvidenceTier.FROZEN_HOLDOUT}
            )
        return tiers == {EvidenceTier.FROZEN_HOLDOUT} and self.contains_empirical_claims


@dataclass(frozen=True)
class GeneratedReportArtifact:
    request: ReportRequest
    path: str
    sha256: str
    renderer_id: str
    renderer_version: str
    source_artifact_ids: tuple[str, ...]
    source_manifest_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        required = (
            self.request.report_id,
            self.path,
            self.sha256,
            self.renderer_id,
            self.renderer_version,
        )
        if not all(required):
            raise ValueError("generated report provenance is required")
        if not self.source_artifact_ids or len(self.source_artifact_ids) != len(
            self.source_manifest_hashes
        ):
            raise ValueError("generated reports require aligned source provenance")
        if self.source_artifact_ids != tuple(source.artifact_id for source in self.request.sources):
            raise ValueError("generated source ids must match the report request")
        if self.source_manifest_hashes != tuple(
            source.manifest_hash for source in self.request.sources
        ):
            raise ValueError("generated manifest hashes must match the report request")

    @property
    def report_id(self) -> str:
        return self.request.report_id

    @property
    def kind(self) -> ReportArtifactKind:
        return self.request.kind

    @property
    def eligible_for_paper(self) -> bool:
        return self.request.paper_eligible


@runtime_checkable
class ReportRenderer(Protocol):
    def render(self, request: ReportRequest, data: object) -> GeneratedReportArtifact:
        """Render only after verifying ``eligible_for_paper == request.paper_eligible``."""
        ...
