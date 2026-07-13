"""Immutable diagnostic results and backend interfaces for daily regimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from .filtering import StateProbabilityBatch
from .hmm import HMMFitSummary
from .vqvae import VQCodeBatch, VQFitSummary


STAGE_OWNER = 7


class DiagnosticStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class DiagnosticFailureCode(str, Enum):
    NON_CONVERGENCE = "non_convergence"
    INVALID_PROBABILITY = "invalid_probability"
    INVALID_COVARIANCE = "invalid_covariance"
    OCCUPANCY_BELOW_FLOOR = "occupancy_below_floor"
    FUTURE_DEPENDENCE = "future_dependence"
    CODEBOOK_COLLAPSE = "codebook_collapse"
    INVALID_SEQUENCE_BOUNDARY = "invalid_sequence_boundary"


@dataclass(frozen=True)
class DiagnosticCheck:
    check_id: str
    status: DiagnosticStatus
    message: str
    value: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.check_id or not self.message:
            raise ValueError("diagnostic check_id and message are required")
        if self.value is not None and not isfinite(self.value):
            raise ValueError("diagnostic value must be finite when present")
        if self.threshold is not None and not isfinite(self.threshold):
            raise ValueError("diagnostic threshold must be finite when present")


@dataclass(frozen=True)
class HMMDiagnosticSummary:
    model_id: str
    normalized_mean_filtered_entropy: float
    soft_state_occupancy: tuple[float, ...]
    expected_dwell_periods: tuple[float, ...]
    effective_state_count: int
    transition_stability_score: float | None
    checks: tuple[DiagnosticCheck, ...]
    failure_codes: tuple[DiagnosticFailureCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not 0.0 <= self.normalized_mean_filtered_entropy <= 1.0:
            raise ValueError("normalized filtered entropy must lie in [0, 1]")
        if not self.soft_state_occupancy:
            raise ValueError("state occupancy is required")
        if any(not isfinite(value) or value < 0.0 for value in self.soft_state_occupancy):
            raise ValueError("state occupancy must be finite and non-negative")
        if abs(sum(self.soft_state_occupancy) - 1.0) > 1e-8:
            raise ValueError("state occupancy must sum to one")
        if len(self.expected_dwell_periods) != len(self.soft_state_occupancy):
            raise ValueError("one dwell-time value is required per state")
        if any(not isfinite(value) or value < 1.0 for value in self.expected_dwell_periods):
            raise ValueError("expected dwell periods must be finite and at least one")
        if not 1 <= self.effective_state_count <= len(self.soft_state_occupancy):
            raise ValueError("effective_state_count is outside the valid range")
        if self.transition_stability_score is not None and (
            not isfinite(self.transition_stability_score)
            or not 0.0 <= self.transition_stability_score <= 1.0
        ):
            raise ValueError("transition_stability_score must lie in [0, 1]")

    @property
    def passed(self) -> bool:
        return not self.failure_codes and all(
            check.status is not DiagnosticStatus.FAIL for check in self.checks
        )


@dataclass(frozen=True)
class CodebookDiagnosticSummary:
    model_id: str
    codebook_size: int
    active_codes: int
    code_perplexity: float
    largest_code_share: float
    reconstruction_loss: float
    checks: tuple[DiagnosticCheck, ...]
    failure_codes: tuple[DiagnosticFailureCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id or self.codebook_size < 2:
            raise ValueError("model_id and a codebook_size of at least two are required")
        if not 1 <= self.active_codes <= self.codebook_size:
            raise ValueError("active_codes must lie in [1, codebook_size]")
        if not isfinite(self.code_perplexity) or not 1.0 <= self.code_perplexity <= self.codebook_size:
            raise ValueError("code_perplexity must lie in [1, codebook_size]")
        if not isfinite(self.largest_code_share) or not 0.0 < self.largest_code_share <= 1.0:
            raise ValueError("largest_code_share must lie in (0, 1]")
        if not isfinite(self.reconstruction_loss) or self.reconstruction_loss < 0.0:
            raise ValueError("reconstruction_loss must be finite and non-negative")

    @property
    def collapsed(self) -> bool:
        return DiagnosticFailureCode.CODEBOOK_COLLAPSE in self.failure_codes

    @property
    def passed(self) -> bool:
        return not self.failure_codes and all(
            check.status is not DiagnosticStatus.FAIL for check in self.checks
        )


@dataclass(frozen=True)
class FuturePerturbationResult:
    cutoff_row_id: str
    baseline_artifact_hash: str
    perturbed_artifact_hash: str
    compared_row_count: int
    identical_through_cutoff: bool

    def __post_init__(self) -> None:
        if not self.cutoff_row_id or not self.baseline_artifact_hash or not self.perturbed_artifact_hash:
            raise ValueError("future-perturbation provenance is required")
        if self.compared_row_count < 1:
            raise ValueError("future-perturbation tests must compare at least one row")
        if self.identical_through_cutoff and (
            self.baseline_artifact_hash != self.perturbed_artifact_hash
        ):
            raise ValueError("identical histories must have identical canonical hashes")


@dataclass(frozen=True)
class DiagnosticBundle:
    run_id: str
    hmm: tuple[HMMDiagnosticSummary, ...] = field(default_factory=tuple)
    codebooks: tuple[CodebookDiagnosticSummary, ...] = field(default_factory=tuple)
    future_perturbation: tuple[FuturePerturbationResult, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")

    @property
    def passed(self) -> bool:
        return (
            bool(self.future_perturbation)
            and all(item.passed for item in self.hmm)
            and all(item.passed for item in self.codebooks)
            and all(item.identical_through_cutoff for item in self.future_perturbation)
        )


@runtime_checkable
class HMMDiagnosticsBackend(Protocol):
    def evaluate(
        self,
        fit: HMMFitSummary,
        probabilities: StateProbabilityBatch,
    ) -> HMMDiagnosticSummary:
        ...


@runtime_checkable
class VQDiagnosticsBackend(Protocol):
    def evaluate(self, fit: VQFitSummary, codes: VQCodeBatch) -> CodebookDiagnosticSummary:
        ...
