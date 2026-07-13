"""Typed contracts for daily-regime hidden Markov models.

This module intentionally contains no estimator implementation.  It defines the
immutable configuration, sequence boundary, fit-result, and estimator contracts
that implementations must satisfy before they can participate in the paper
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Generic, Protocol, TypeVar, runtime_checkable


STAGE_OWNER = 7


class HMMFamily(str, Enum):
    """Preregistered latent-state model families."""

    GAUSSIAN = "gaussian_hmm"
    GAUSSIAN_MIXTURE = "gmm_hmm"
    STUDENT_T = "student_t_hmm"


class CovarianceType(str, Enum):
    DIAGONAL = "diagonal"
    FULL = "full"


class HMMFailureCode(str, Enum):
    NON_CONVERGENCE = "non_convergence"
    INVALID_PROBABILITY = "invalid_probability"
    INVALID_COVARIANCE = "invalid_covariance"
    OCCUPANCY_BELOW_FLOOR = "occupancy_below_floor"
    NONFINITE_LIKELIHOOD = "nonfinite_likelihood"
    FUTURE_DEPENDENCE = "future_dependence"


@dataclass(frozen=True)
class HMMConfig:
    """Frozen model grid entry.

    The configuration represents one trial, not a search space.  Restarts are
    deterministic offsets from ``random_seed`` and every attempted restart must
    appear in the eventual trial ledger.
    """

    family: HMMFamily
    n_states: int
    n_mixtures: int = 1
    covariance_type: CovarianceType = CovarianceType.DIAGONAL
    n_restarts: int = 20
    max_iterations: int = 500
    tolerance: float = 1e-6
    covariance_floor: float = 1e-6
    minimum_state_occupancy: float = 0.05
    random_seed: int = 42
    fit_scope: str = "training_only"

    def __post_init__(self) -> None:
        if not isinstance(self.family, HMMFamily):
            raise TypeError("family must be an HMMFamily")
        if not isinstance(self.covariance_type, CovarianceType):
            raise TypeError("covariance_type must be a CovarianceType")
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2")
        if self.n_mixtures < 1:
            raise ValueError("n_mixtures must be positive")
        if self.family is not HMMFamily.GAUSSIAN_MIXTURE and self.n_mixtures != 1:
            raise ValueError("n_mixtures must equal 1 outside the GMM-HMM family")
        if self.n_restarts < 1 or self.max_iterations < 1:
            raise ValueError("n_restarts and max_iterations must be positive")
        if not isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not isfinite(self.covariance_floor) or self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be finite and positive")
        if not 0.0 < self.minimum_state_occupancy < 1.0:
            raise ValueError("minimum_state_occupancy must lie strictly between 0 and 1")
        if self.n_states * self.minimum_state_occupancy > 1.0:
            raise ValueError("occupancy floor is infeasible for the requested state count")
        if self.fit_scope != "training_only":
            raise ValueError("HMM fitting is restricted to training_only scope")


@dataclass(frozen=True)
class HMMSequence:
    """One product-safe, contiguous sequence supplied to an HMM.

    A sequence has exactly one product and one declared segment.  Discontinuity
    handling belongs upstream: a missing interval creates another sequence
    rather than an artificial transition.
    """

    sequence_id: str
    product_id: str
    row_ids: tuple[str, ...]
    input_columns: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.sequence_id or not self.product_id:
            raise ValueError("sequence_id and product_id are required")
        if not self.input_columns or len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("input_columns must be non-empty and unique")
        if not self.row_ids or len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("row_ids must be non-empty and unique within a sequence")
        if len(self.values) != len(self.row_ids):
            raise ValueError("one feature row is required for every row_id")
        width = len(self.input_columns)
        for row in self.values:
            if len(row) != width:
                raise ValueError("all feature rows must match input_columns")
            if any(not isfinite(float(value)) for value in row):
                raise ValueError("nonfinite features may not enter an HMM")


@dataclass(frozen=True)
class HMMTrainingBatch:
    """Training sequences with explicit feature and fold provenance."""

    fold_id: str
    feature_set_id: str
    preprocessing_artifact_hash: str
    sequences: tuple[HMMSequence, ...]

    def __post_init__(self) -> None:
        if not self.fold_id or not self.feature_set_id:
            raise ValueError("fold_id and feature_set_id are required")
        if not self.preprocessing_artifact_hash:
            raise ValueError("preprocessing_artifact_hash is required")
        if not self.sequences:
            raise ValueError("at least one training sequence is required")
        columns = self.sequences[0].input_columns
        sequence_ids: set[str] = set()
        row_ids: set[str] = set()
        for sequence in self.sequences:
            if sequence.input_columns != columns:
                raise ValueError("all HMM sequences must use identical ordered columns")
            if sequence.sequence_id in sequence_ids:
                raise ValueError("sequence_id values must be unique")
            sequence_ids.add(sequence.sequence_id)
            overlap = row_ids.intersection(sequence.row_ids)
            if overlap:
                raise ValueError("row_ids may not occur in more than one sequence")
            row_ids.update(sequence.row_ids)

    @property
    def n_observations(self) -> int:
        return sum(len(sequence.row_ids) for sequence in self.sequences)


@dataclass(frozen=True)
class HMMRestartResult:
    restart_index: int
    seed: int
    converged: bool
    iterations: int
    training_log_likelihood: float | None
    failure_codes: tuple[HMMFailureCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.restart_index < 0 or self.iterations < 0:
            raise ValueError("restart_index and iterations must be non-negative")
        if self.training_log_likelihood is not None and not isfinite(
            self.training_log_likelihood
        ):
            raise ValueError("training_log_likelihood must be finite when present")
        if self.converged and self.failure_codes:
            raise ValueError("a converged restart cannot also carry failure codes")


@dataclass(frozen=True)
class HMMFitSummary:
    """Auditable result of fitting one frozen HMM configuration."""

    model_id: str
    fold_id: str
    feature_set_id: str
    selected_restart_index: int | None
    restarts: tuple[HMMRestartResult, ...]
    soft_state_occupancy: tuple[float, ...]
    training_rows_hash: str
    parameter_hash: str | None
    failure_codes: tuple[HMMFailureCode, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id or not self.fold_id or not self.feature_set_id:
            raise ValueError("model, fold, and feature-set identifiers are required")
        if not self.restarts:
            raise ValueError("the complete restart ledger is required")
        restart_indices = [item.restart_index for item in self.restarts]
        if len(set(restart_indices)) != len(restart_indices):
            raise ValueError("restart indices must be unique")
        if not self.training_rows_hash:
            raise ValueError("training_rows_hash is required")
        if any(not isfinite(value) or value < 0.0 for value in self.soft_state_occupancy):
            raise ValueError("state occupancies must be finite and non-negative")
        if self.soft_state_occupancy and abs(sum(self.soft_state_occupancy) - 1.0) > 1e-8:
            raise ValueError("soft state occupancies must sum to one")
        if self.selected_restart_index is not None:
            selected = [
                item for item in self.restarts if item.restart_index == self.selected_restart_index
            ]
            if len(selected) != 1 or not selected[0].converged:
                raise ValueError("selected restart must identify one converged ledger entry")
            if not self.parameter_hash:
                raise ValueError("a selected fitted model requires parameter_hash")
        elif self.parameter_hash is not None:
            raise ValueError("a failed fit cannot expose a selected parameter hash")
        elif not self.failure_codes:
            raise ValueError("an unsuccessful fit must record failure codes")

    @property
    def successful(self) -> bool:
        return self.selected_restart_index is not None and not self.failure_codes


ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class HMMFitResult(Generic[ModelT]):
    """In-memory fitted model plus its immutable audit summary."""

    config: HMMConfig
    model: ModelT | None
    summary: HMMFitSummary

    def __post_init__(self) -> None:
        if len(self.summary.restarts) != self.config.n_restarts:
            raise ValueError("restart ledger length must match n_restarts")
        if {item.restart_index for item in self.summary.restarts} != set(
            range(self.config.n_restarts)
        ):
            raise ValueError("restart ledger must cover indices 0..n_restarts-1")
        if self.summary.selected_restart_index is None:
            if self.model is not None or self.summary.soft_state_occupancy:
                raise ValueError("a completely failed fit has no model or state occupancy")
            return
        if self.model is None:
            raise ValueError("a selected restart requires an in-memory model")
        if len(self.summary.soft_state_occupancy) != self.config.n_states:
            raise ValueError("fit summary occupancy count must match n_states")
        if any(
            value < self.config.minimum_state_occupancy
            for value in self.summary.soft_state_occupancy
        ) and HMMFailureCode.OCCUPANCY_BELOW_FLOOR not in self.summary.failure_codes:
            raise ValueError("sub-floor occupancy must be recorded as a failed trial")


@runtime_checkable
class HMMEstimator(Protocol[ModelT]):
    """Runtime-checkable interface implemented by every HMM backend."""

    def fit(self, batch: HMMTrainingBatch, config: HMMConfig) -> HMMFitResult[ModelT]:
        ...

    def log_emission_probabilities(
        self,
        fitted: HMMFitResult[ModelT],
        sequence: HMMSequence,
    ) -> tuple[tuple[float, ...], ...]:
        """Return one row of log-emission values per observation and state."""
        ...
