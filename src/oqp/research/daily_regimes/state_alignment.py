"""Training-only state-label alignment contracts.

Latent-state labels are arbitrary.  Alignment exists solely to make refits
comparable in diagnostics and figures; it must never alter likelihoods,
external targets, or model selection scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable


STAGE_OWNER = 7


class AlignmentMetric(str, Enum):
    EMISSION_MEAN_SCALED_L2 = "emission_mean_scaled_l2"
    SYMMETRIC_GAUSSIAN_KL = "symmetric_gaussian_kl"


class AlignmentTieBreak(str, Enum):
    LEXICOGRAPHIC_STATE_INDEX = "lexicographic_state_index"


@dataclass(frozen=True)
class StateAlignmentConfig:
    metric: AlignmentMetric = AlignmentMetric.EMISSION_MEAN_SCALED_L2
    feature_weights: tuple[float, ...] = ()
    tie_break: AlignmentTieBreak = AlignmentTieBreak.LEXICOGRAPHIC_STATE_INDEX
    fit_scope: str = "training_only"

    def __post_init__(self) -> None:
        if not isinstance(self.metric, AlignmentMetric):
            raise TypeError("metric must be an AlignmentMetric")
        if not isinstance(self.tie_break, AlignmentTieBreak):
            raise TypeError("tie_break must be an AlignmentTieBreak")
        if self.feature_weights and any(
            not isfinite(value) or value <= 0.0 for value in self.feature_weights
        ):
            raise ValueError("feature weights must be finite and positive")
        if self.fit_scope != "training_only":
            raise ValueError("state alignment is restricted to training-only summaries")


@dataclass(frozen=True)
class StateSignature:
    """Training-only emission summary for one arbitrary model state."""

    state_index: int
    feature_names: tuple[str, ...]
    location: tuple[float, ...]
    scale: tuple[float, ...]
    training_rows_hash: str

    def __post_init__(self) -> None:
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        width = len(self.feature_names)
        if len(self.location) != width or len(self.scale) != width:
            raise ValueError("location and scale must match feature_names")
        if any(not isfinite(value) for value in self.location):
            raise ValueError("state locations must be finite")
        if any(not isfinite(value) or value <= 0.0 for value in self.scale):
            raise ValueError("state scales must be finite and positive")
        if not self.training_rows_hash:
            raise ValueError("training_rows_hash is required")


@dataclass(frozen=True)
class StateAlignmentInput:
    reference_fit_id: str
    candidate_fit_id: str
    reference: tuple[StateSignature, ...]
    candidate: tuple[StateSignature, ...]
    information_scope: str = "training_only"

    def __post_init__(self) -> None:
        if not self.reference_fit_id or not self.candidate_fit_id:
            raise ValueError("reference and candidate fit identifiers are required")
        if not self.reference or len(self.reference) != len(self.candidate):
            raise ValueError("reference and candidate must have the same nonzero state count")
        if self.information_scope != "training_only":
            raise ValueError("alignment input may contain training information only")
        reference_indices = {item.state_index for item in self.reference}
        candidate_indices = {item.state_index for item in self.candidate}
        expected = set(range(len(self.reference)))
        if reference_indices != expected or candidate_indices != expected:
            raise ValueError("state indices must each form 0..K-1")
        columns = self.reference[0].feature_names
        if any(item.feature_names != columns for item in (*self.reference, *self.candidate)):
            raise ValueError("all state signatures must use identical ordered features")


@dataclass(frozen=True)
class StatePermutation:
    """Candidate-state index to canonical reference-state index."""

    candidate_to_reference: tuple[int, ...]

    def __post_init__(self) -> None:
        expected = set(range(len(self.candidate_to_reference)))
        if set(self.candidate_to_reference) != expected:
            raise ValueError("candidate_to_reference must be a complete permutation")


@dataclass(frozen=True)
class StateAlignmentResult:
    reference_fit_id: str
    candidate_fit_id: str
    permutation: StatePermutation
    cost_matrix: tuple[tuple[float, ...], ...]
    total_cost: float
    reference_training_rows_hash: str
    candidate_training_rows_hash: str

    def __post_init__(self) -> None:
        width = len(self.permutation.candidate_to_reference)
        if len(self.cost_matrix) != width or any(len(row) != width for row in self.cost_matrix):
            raise ValueError("cost_matrix must be square and match the state count")
        if any(not isfinite(value) or value < 0.0 for row in self.cost_matrix for value in row):
            raise ValueError("alignment costs must be finite and non-negative")
        if not isfinite(self.total_cost) or self.total_cost < 0.0:
            raise ValueError("total_cost must be finite and non-negative")
        if not self.reference_training_rows_hash or not self.candidate_training_rows_hash:
            raise ValueError("training-row hashes are required")


@runtime_checkable
class StateAligner(Protocol):
    def align(
        self,
        inputs: StateAlignmentInput,
        config: StateAlignmentConfig,
    ) -> StateAlignmentResult:
        ...
