"""Deterministic state-label alignment for fitted hidden Markov models.

HMM state indices are arbitrary: two equivalent fits may assign different
integer labels to the same emission regimes.  This module aligns candidate
states to a reference fit using training-only emission summaries.  Alignment
is a diagnostic/canonicalization operation; it must not alter likelihoods,
targets, or model-selection scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import HMMFamily
from .fitted import FittedDiagonalHMM
from .serialization import sha256_json


_HEX_DIGITS = frozenset("0123456789abcdef")


class AlignmentMetric(str, Enum):
    """Supported distances between marginal state-emission signatures."""

    STANDARDIZED_MEAN_SQUARED_DISTANCE = "standardized_mean_squared_distance"
    SYMMETRIC_GAUSSIAN_KL = "symmetric_gaussian_kl"


class AlignmentTieBreak(str, Enum):
    """Deterministic policy applied to assignments inside the tie band."""

    LEXICOGRAPHIC_STATE_INDEX = "lexicographic_state_index"


@dataclass(frozen=True, slots=True)
class StateAlignmentConfig:
    """Metric and numerical policy for one alignment decision."""

    metric: AlignmentMetric = AlignmentMetric.STANDARDIZED_MEAN_SQUARED_DISTANCE
    feature_weights: tuple[float, ...] = ()
    variance_floor: float = 1e-6
    tie_break: AlignmentTieBreak = AlignmentTieBreak.LEXICOGRAPHIC_STATE_INDEX
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.metric, AlignmentMetric):
            raise TypeError("metric must be an AlignmentMetric")
        if not isinstance(self.tie_break, AlignmentTieBreak):
            raise TypeError("tie_break must be an AlignmentTieBreak")
        weights = tuple(float(value) for value in self.feature_weights)
        if weights and any(not isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("feature_weights must be finite and positive")
        if not isfinite(self.variance_floor) or self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be finite and positive")
        if not isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be finite and non-negative")
        object.__setattr__(self, "feature_weights", weights)

    def state_dict(self) -> dict[str, object]:
        """Return the JSON-safe policy included in alignment provenance."""

        return {
            "metric": self.metric.value,
            "feature_weights": list(self.feature_weights),
            "variance_floor_hex": float(self.variance_floor).hex(),
            "tie_break": self.tie_break.value,
            "tie_tolerance_hex": float(self.tie_tolerance).hex(),
        }


@dataclass(frozen=True, slots=True)
class StateSignature:
    """Training-data emission summary for one arbitrary state index.

    ``scale`` is the marginal emission standard deviation, regardless of the
    model family's internal scale parameterization.
    """

    state_index: int
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    location: tuple[float, ...]
    scale: tuple[float, ...]
    training_data_sha256: str

    def __post_init__(self) -> None:
        if type(self.state_index) is not int or self.state_index < 0:
            raise ValueError("state_index must be a non-negative integer")
        features = tuple(self.feature_names)
        if not features or any(type(value) is not str or not value for value in features):
            raise ValueError("feature_names must contain non-empty strings")
        if len(set(features)) != len(features):
            raise ValueError("feature_names must be unique")
        _require_sha256(self.feature_schema_sha256, "feature_schema_sha256")
        _require_sha256(self.training_data_sha256, "training_data_sha256")
        location = tuple(float(value) for value in self.location)
        scale = tuple(float(value) for value in self.scale)
        if len(location) != len(features) or len(scale) != len(features):
            raise ValueError("location and scale must match feature_names")
        if any(not isfinite(value) for value in location):
            raise ValueError("state locations must be finite")
        if any(not isfinite(value) or value <= 0.0 for value in scale):
            raise ValueError("state scales must be finite and positive")
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True, slots=True)
class StateAlignmentInput:
    """Reference and candidate signatures derived only from training data."""

    reference_model_id: str
    candidate_model_id: str
    reference: tuple[StateSignature, ...]
    candidate: tuple[StateSignature, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.reference_model_id, "reference_model_id")
        _require_identifier(self.candidate_model_id, "candidate_model_id")
        reference = tuple(self.reference)
        candidate = tuple(self.candidate)
        if not reference or len(reference) != len(candidate):
            raise ValueError("reference and candidate must have the same state count")
        if any(not isinstance(item, StateSignature) for item in (*reference, *candidate)):
            raise TypeError("reference and candidate must contain StateSignature objects")
        expected = set(range(len(reference)))
        if {item.state_index for item in reference} != expected:
            raise ValueError("reference state indices must form 0..K-1")
        if {item.state_index for item in candidate} != expected:
            raise ValueError("candidate state indices must form 0..K-1")
        feature_names = reference[0].feature_names
        schema_hash = reference[0].feature_schema_sha256
        if any(item.feature_names != feature_names for item in (*reference, *candidate)):
            raise ValueError("all signatures must use identical ordered features")
        if any(
            item.feature_schema_sha256 != schema_hash
            for item in (*reference, *candidate)
        ):
            raise ValueError("all signatures must use the same authenticated schema")
        if len({item.training_data_sha256 for item in reference}) != 1:
            raise ValueError("reference signatures must share one training-data hash")
        if len({item.training_data_sha256 for item in candidate}) != 1:
            raise ValueError("candidate signatures must share one training-data hash")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "candidate", candidate)


@dataclass(frozen=True, slots=True)
class StatePermutation:
    """Mapping from candidate-state indices to reference-state indices."""

    candidate_to_reference: tuple[int, ...]

    def __post_init__(self) -> None:
        mapping = tuple(self.candidate_to_reference)
        if not mapping:
            raise ValueError("candidate_to_reference cannot be empty")
        if any(type(value) is not int for value in mapping):
            raise TypeError("candidate_to_reference must contain integers")
        if set(mapping) != set(range(len(mapping))):
            raise ValueError("candidate_to_reference must be a complete permutation")
        object.__setattr__(self, "candidate_to_reference", mapping)

    @property
    def reference_to_candidate(self) -> tuple[int, ...]:
        """Return the inverse mapping used to reorder candidate arrays."""

        inverse = [0] * len(self.candidate_to_reference)
        for candidate_index, reference_index in enumerate(
            self.candidate_to_reference
        ):
            inverse[reference_index] = candidate_index
        return tuple(inverse)


@dataclass(frozen=True, slots=True)
class StateAlignmentResult:
    """Immutable alignment decision with metric and training provenance."""

    reference_model_id: str
    candidate_model_id: str
    permutation: StatePermutation
    cost_matrix: tuple[tuple[float, ...], ...]
    total_cost: float
    reference_training_data_sha256: str
    candidate_training_data_sha256: str
    config: StateAlignmentConfig

    def __post_init__(self) -> None:
        _require_identifier(self.reference_model_id, "reference_model_id")
        _require_identifier(self.candidate_model_id, "candidate_model_id")
        if not isinstance(self.permutation, StatePermutation):
            raise TypeError("permutation must be a StatePermutation")
        if not isinstance(self.config, StateAlignmentConfig):
            raise TypeError("config must be a StateAlignmentConfig")
        _require_sha256(
            self.reference_training_data_sha256,
            "reference_training_data_sha256",
        )
        _require_sha256(
            self.candidate_training_data_sha256,
            "candidate_training_data_sha256",
        )
        width = len(self.permutation.candidate_to_reference)
        matrix = tuple(tuple(float(value) for value in row) for row in self.cost_matrix)
        if len(matrix) != width or any(len(row) != width for row in matrix):
            raise ValueError("cost_matrix must be square and match the state count")
        if any(
            not isfinite(value) or value < 0.0 for row in matrix for value in row
        ):
            raise ValueError("alignment costs must be finite and non-negative")
        if not isfinite(self.total_cost) or self.total_cost < 0.0:
            raise ValueError("total_cost must be finite and non-negative")
        assigned_cost = fsum(
            matrix[candidate_index][reference_index]
            for candidate_index, reference_index in enumerate(
                self.permutation.candidate_to_reference
            )
        )
        tolerance = 1e-10 * max(1.0, assigned_cost, self.total_cost)
        if abs(assigned_cost - self.total_cost) > tolerance:
            raise ValueError("total_cost must match the declared permutation")
        object.__setattr__(self, "cost_matrix", matrix)
        object.__setattr__(self, "total_cost", float(self.total_cost))

    @property
    def alignment_sha256(self) -> str:
        """Authenticate the decision, costs, policy, and training provenance."""

        return sha256_json(
            {
                "reference_model_id": self.reference_model_id,
                "candidate_model_id": self.candidate_model_id,
                "candidate_to_reference": list(
                    self.permutation.candidate_to_reference
                ),
                "cost_matrix_hex": [
                    [float(value).hex() for value in row] for row in self.cost_matrix
                ],
                "total_cost_hex": float(self.total_cost).hex(),
                "reference_training_data_sha256": (
                    self.reference_training_data_sha256
                ),
                "candidate_training_data_sha256": (
                    self.candidate_training_data_sha256
                ),
                "config": self.config.state_dict(),
            }
        )


@runtime_checkable
class StateAligner(Protocol):
    """Interface for deterministic state-label alignment backends."""

    def align(
        self,
        inputs: StateAlignmentInput,
        config: StateAlignmentConfig,
    ) -> StateAlignmentResult:
        """Align candidate state indices to the reference state indices."""
        ...


@dataclass(frozen=True, slots=True)
class HungarianStateAligner:
    """Minimum-cost assignment with deterministic lexicographic ties."""

    def align(
        self,
        inputs: StateAlignmentInput,
        config: StateAlignmentConfig,
    ) -> StateAlignmentResult:
        if not isinstance(inputs, StateAlignmentInput):
            raise TypeError("inputs must be a StateAlignmentInput")
        if not isinstance(config, StateAlignmentConfig):
            raise TypeError("config must be a StateAlignmentConfig")
        costs = build_state_alignment_cost_matrix(inputs, config)
        permutation = deterministic_hungarian_assignment(
            costs,
            tie_tolerance=config.tie_tolerance,
        )
        total_cost = fsum(
            costs[candidate_index][reference_index]
            for candidate_index, reference_index in enumerate(
                permutation.candidate_to_reference
            )
        )
        reference = sorted(inputs.reference, key=lambda item: item.state_index)
        candidate = sorted(inputs.candidate, key=lambda item: item.state_index)
        return StateAlignmentResult(
            reference_model_id=inputs.reference_model_id,
            candidate_model_id=inputs.candidate_model_id,
            permutation=permutation,
            cost_matrix=costs,
            total_cost=total_cost,
            reference_training_data_sha256=reference[0].training_data_sha256,
            candidate_training_data_sha256=candidate[0].training_data_sha256,
            config=config,
        )


def align_states(
    inputs: StateAlignmentInput,
    config: StateAlignmentConfig | None = None,
) -> StateAlignmentResult:
    """Align candidate states using deterministic training-emission summaries."""

    return HungarianStateAligner().align(inputs, config or StateAlignmentConfig())


def state_signatures_from_fitted_hmm(
    model: FittedDiagonalHMM,
    *,
    training_data_sha256: str,
) -> tuple[StateSignature, ...]:
    """Build marginal mean/std signatures from a shared fitted HMM.

    The caller must provide the authenticated hash of the training batch.  It
    is available as ``HMMTrainingResult.training_data_sha256`` and is kept
    separate from the fitted parameter artifact by the shared model contract.
    """

    if not isinstance(model, FittedDiagonalHMM):
        raise TypeError("model must be a FittedDiagonalHMM")
    _require_sha256(training_data_sha256, "training_data_sha256")
    weights = np.asarray(model.mixture_weights, dtype=np.float64)
    means = np.asarray(model.means, dtype=np.float64)
    component_variances = np.asarray(model.diagonal_scales, dtype=np.float64)
    if model.family is HMMFamily.STUDENT_T:
        degrees = model.student_t_degrees_of_freedom
        if degrees is None or degrees <= 2.0:  # guarded by fitted-model validation
            raise ValueError("Student-t signatures require finite variance")
        component_variances = component_variances * (degrees / (degrees - 2.0))
    marginal_means = np.sum(weights[:, :, None] * means, axis=1)
    second_moments = np.sum(
        weights[:, :, None] * (component_variances + means**2),
        axis=1,
    )
    marginal_variances = second_moments - marginal_means**2
    if not np.isfinite(marginal_variances).all():
        raise ValueError("marginal emission variances must be finite")
    marginal_scales = np.sqrt(
        np.maximum(marginal_variances, np.finfo(np.float64).tiny)
    )
    schema_hash = model.feature_schema.schema_sha256
    if schema_hash is None:  # guarded by OrderedFeatureSchema
        raise RuntimeError("fitted model has no authenticated feature schema")
    return tuple(
        StateSignature(
            state_index=state_index,
            feature_names=model.feature_schema.feature_names,
            feature_schema_sha256=schema_hash,
            location=tuple(float(value) for value in marginal_means[state_index]),
            scale=tuple(float(value) for value in marginal_scales[state_index]),
            training_data_sha256=training_data_sha256,
        )
        for state_index in range(model.n_states)
    )


def build_state_alignment_cost_matrix(
    inputs: StateAlignmentInput,
    config: StateAlignmentConfig,
) -> tuple[tuple[float, ...], ...]:
    """Return candidate-row/reference-column emission distances."""

    if not isinstance(inputs, StateAlignmentInput):
        raise TypeError("inputs must be a StateAlignmentInput")
    if not isinstance(config, StateAlignmentConfig):
        raise TypeError("config must be a StateAlignmentConfig")
    reference = sorted(inputs.reference, key=lambda item: item.state_index)
    candidate = sorted(inputs.candidate, key=lambda item: item.state_index)
    weights = _normalized_feature_weights(
        config.feature_weights,
        feature_count=len(reference[0].feature_names),
    )
    matrix: list[tuple[float, ...]] = []
    for candidate_state in candidate:
        row: list[float] = []
        for reference_state in reference:
            cost = _state_signature_cost(
                reference_state,
                candidate_state,
                metric=config.metric,
                normalized_weights=weights,
                variance_floor=config.variance_floor,
            )
            if not isfinite(cost) or cost < 0.0:
                raise ValueError("alignment metric produced an invalid cost")
            row.append(float(cost))
        matrix.append(tuple(row))
    return tuple(matrix)


def deterministic_hungarian_assignment(
    cost_matrix: Sequence[Sequence[float]],
    *,
    tie_tolerance: float = 1e-12,
) -> StatePermutation:
    """Return the lexicographically first assignment inside the tie band.

    One Hungarian solve establishes the optimum.  Candidate rows are then
    fixed from left to right; each proposed reference column is accepted only
    if a Hungarian solve of the remaining subproblem can still attain the
    optimum within ``tie_tolerance``.  This is deterministic without factorial
    enumeration as the number of states grows.
    """

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] < 1 or costs.shape[0] != costs.shape[1]:
        raise ValueError("alignment cost matrix must be non-empty and square")
    if not np.isfinite(costs).all() or (costs < 0.0).any():
        raise ValueError("alignment cost matrix must be finite and non-negative")
    if not isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    rows, columns = linear_sum_assignment(costs)
    optimum_mapping = [0] * costs.shape[0]
    for row, column in zip(rows, columns):
        optimum_mapping[int(row)] = int(column)
    optimum = fsum(
        float(costs[row, optimum_mapping[row]]) for row in range(costs.shape[0])
    )
    available = list(range(costs.shape[0]))
    mapping: list[int] = []
    for candidate_index in range(costs.shape[0]):
        selected: int | None = None
        for reference_index in available:
            remaining_columns = [
                value for value in available if value != reference_index
            ]
            completion: list[int] = []
            if remaining_columns:
                remaining_rows = np.arange(candidate_index + 1, costs.shape[0])
                subproblem = costs[np.ix_(remaining_rows, remaining_columns)]
                sub_rows, sub_columns = linear_sum_assignment(subproblem)
                completion = [0] * len(remaining_columns)
                for row, column in zip(sub_rows, sub_columns):
                    completion[int(row)] = remaining_columns[int(column)]
            trial_mapping = [*mapping, reference_index, *completion]
            trial_cost = fsum(
                float(costs[row, trial_mapping[row]])
                for row in range(costs.shape[0])
            )
            if abs(trial_cost - optimum) <= tie_tolerance:
                selected = reference_index
                break
        if selected is None:
            raise RuntimeError("unable to reproduce the Hungarian optimum")
        mapping.append(selected)
        available.remove(selected)
    return StatePermutation(candidate_to_reference=tuple(mapping))


ValueT = TypeVar("ValueT")


def reorder_candidate_values_to_reference(
    values: Sequence[ValueT],
    permutation: StatePermutation,
) -> tuple[ValueT, ...]:
    """Reorder candidate-indexed values into reference-state order."""

    items = tuple(values)
    if len(items) != len(permutation.candidate_to_reference):
        raise ValueError("candidate value width must match the state permutation")
    return tuple(items[index] for index in permutation.reference_to_candidate)


def reorder_candidate_probabilities_to_reference(
    probabilities: Sequence[float],
    permutation: StatePermutation,
    *,
    tolerance: float = 1e-10,
) -> tuple[float, ...]:
    """Validate and reorder one candidate-state probability simplex."""

    values = np.asarray(probabilities, dtype=np.float64)
    width = len(permutation.candidate_to_reference)
    if values.shape != (width,):
        raise ValueError("probability width must match the state permutation")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    return reorder_candidate_values_to_reference(
        tuple(float(value) for value in values / total),
        permutation,
    )


def reorder_candidate_transition_matrix_to_reference(
    transition_matrix: Sequence[Sequence[float]],
    permutation: StatePermutation,
    *,
    tolerance: float = 1e-10,
) -> tuple[tuple[float, ...], ...]:
    """Reorder both axes of a candidate-state transition matrix."""

    values = np.asarray(transition_matrix, dtype=np.float64)
    width = len(permutation.candidate_to_reference)
    if values.shape != (width, width):
        raise ValueError("transition matrix must be square and match the permutation")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("transition probabilities must be finite and lie in [0, 1]")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("every transition row must sum to one")
    order = permutation.reference_to_candidate
    reordered = values[np.ix_(order, order)]
    reordered /= reordered.sum(axis=1, keepdims=True)
    return tuple(tuple(float(value) for value in row) for row in reordered)


def _normalized_feature_weights(
    feature_weights: tuple[float, ...],
    *,
    feature_count: int,
) -> np.ndarray:
    if feature_weights and len(feature_weights) != feature_count:
        raise ValueError("feature_weights must match the ordered feature count")
    weights = np.asarray(
        feature_weights or (1.0,) * feature_count,
        dtype=np.float64,
    )
    return weights / weights.sum()


def _state_signature_cost(
    reference: StateSignature,
    candidate: StateSignature,
    *,
    metric: AlignmentMetric,
    normalized_weights: np.ndarray,
    variance_floor: float,
) -> float:
    reference_location = np.asarray(reference.location, dtype=np.float64)
    candidate_location = np.asarray(candidate.location, dtype=np.float64)
    reference_scale = np.asarray(reference.scale, dtype=np.float64)
    candidate_scale = np.asarray(candidate.scale, dtype=np.float64)
    difference = candidate_location - reference_location
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if metric is AlignmentMetric.STANDARDIZED_MEAN_SQUARED_DISTANCE:
            denominator = (
                0.5 * (reference_scale**2 + candidate_scale**2) + variance_floor
            )
            return float(np.sum(normalized_weights * difference**2 / denominator))
        if metric is AlignmentMetric.SYMMETRIC_GAUSSIAN_KL:
            reference_variance = reference_scale**2
            candidate_variance = candidate_scale**2
            per_feature = 0.25 * (
                reference_variance / candidate_variance
                + candidate_variance / reference_variance
                + difference**2
                * (1.0 / reference_variance + 1.0 / candidate_variance)
                - 2.0
            )
            return max(0.0, float(np.sum(normalized_weights * per_feature)))
    raise ValueError(f"unsupported alignment metric: {metric!r}")


def _require_identifier(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "AlignmentMetric",
    "AlignmentTieBreak",
    "HungarianStateAligner",
    "StateAligner",
    "StateAlignmentConfig",
    "StateAlignmentInput",
    "StateAlignmentResult",
    "StatePermutation",
    "StateSignature",
    "align_states",
    "build_state_alignment_cost_matrix",
    "deterministic_hungarian_assignment",
    "reorder_candidate_probabilities_to_reference",
    "reorder_candidate_transition_matrix_to_reference",
    "reorder_candidate_values_to_reference",
    "state_signatures_from_fitted_hmm",
]
