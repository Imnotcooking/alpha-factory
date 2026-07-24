from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from oqp.contracts.regime_state import OrderedFeatureSchema
from oqp.research.ml.regimes import (
    AlignmentMetric,
    HMMFamily,
    StateAlignmentConfig,
    StateAlignmentInput,
    StatePermutation,
    StateSignature,
    align_states,
    build_state_alignment_cost_matrix,
    deterministic_hungarian_assignment,
    reorder_candidate_probabilities_to_reference,
    reorder_candidate_transition_matrix_to_reference,
    reorder_candidate_values_to_reference,
    state_signatures_from_fitted_hmm,
)
from oqp.research.ml.regimes.fitted import FittedDiagonalHMM


REFERENCE_DATA_SHA256 = "a" * 64
CANDIDATE_DATA_SHA256 = "b" * 64
SCHEMA = OrderedFeatureSchema(
    schema_id="alignment-test",
    feature_names=("x0", "x1"),
)


def _model(
    family: HMMFamily,
    *,
    means: tuple[tuple[tuple[float, ...], ...], ...],
    diagonal_scales: tuple[tuple[tuple[float, ...], ...], ...],
    mixture_weights: tuple[tuple[float, ...], ...] | None = None,
    degrees_of_freedom: float | None = None,
) -> FittedDiagonalHMM:
    state_count = len(means)
    return FittedDiagonalHMM(
        model_id=f"alignment-{family.value}",
        family=family,
        feature_schema=SCHEMA,
        initial_probabilities=np.full(state_count, 1.0 / state_count),
        transition_matrix=np.full((state_count, state_count), 1.0 / state_count),
        means=np.asarray(means),
        diagonal_scales=np.asarray(diagonal_scales),
        mixture_weights=(
            None if mixture_weights is None else np.asarray(mixture_weights)
        ),
        student_t_degrees_of_freedom=degrees_of_freedom,
    )


def _signature(
    state_index: int,
    location: tuple[float, ...],
    *,
    scale: tuple[float, ...] | None = None,
    training_data_sha256: str = REFERENCE_DATA_SHA256,
) -> StateSignature:
    schema_hash = SCHEMA.schema_sha256
    assert schema_hash is not None
    return StateSignature(
        state_index=state_index,
        feature_names=SCHEMA.feature_names,
        feature_schema_sha256=schema_hash,
        location=location,
        scale=scale or (1.0,) * len(location),
        training_data_sha256=training_data_sha256,
    )


def _inputs(
    reference: tuple[StateSignature, ...],
    candidate: tuple[StateSignature, ...],
) -> StateAlignmentInput:
    return StateAlignmentInput(
        reference_model_id="reference-model",
        candidate_model_id="candidate-model",
        reference=reference,
        candidate=candidate,
    )


def test_gaussian_signatures_convert_variances_to_standard_deviations() -> None:
    model = _model(
        HMMFamily.GAUSSIAN,
        means=(((1.0, -2.0),), ((3.0, 4.0),)),
        diagonal_scales=(((4.0, 9.0),), ((0.25, 16.0),)),
    )

    signatures = state_signatures_from_fitted_hmm(
        model,
        training_data_sha256=REFERENCE_DATA_SHA256,
    )

    assert tuple(item.location for item in signatures) == (
        (1.0, -2.0),
        (3.0, 4.0),
    )
    np.testing.assert_allclose(
        tuple(item.scale for item in signatures),
        ((2.0, 3.0), (0.5, 4.0)),
    )
    assert all(item.feature_schema_sha256 == SCHEMA.schema_sha256 for item in signatures)
    assert all(item.training_data_sha256 == REFERENCE_DATA_SHA256 for item in signatures)


def test_student_t_signatures_use_finite_marginal_variance() -> None:
    model = _model(
        HMMFamily.STUDENT_T,
        means=(((0.0, 1.0),), ((2.0, -1.0),)),
        diagonal_scales=(((3.0, 12.0),), ((0.75, 6.75),)),
        degrees_of_freedom=8.0,
    )

    signatures = state_signatures_from_fitted_hmm(
        model,
        training_data_sha256=REFERENCE_DATA_SHA256,
    )

    np.testing.assert_allclose(
        tuple(item.scale for item in signatures),
        ((2.0, 4.0), (1.0, 3.0)),
    )


def test_gmm_signatures_include_within_and_between_component_variance() -> None:
    model = _model(
        HMMFamily.GAUSSIAN_MIXTURE,
        mixture_weights=((0.25, 0.75), (0.5, 0.5)),
        means=(((-2.0, 1.0), (2.0, 5.0)), ((0.0, -3.0), (4.0, 1.0))),
        diagonal_scales=(
            ((1.0, 4.0), (9.0, 16.0)),
            ((4.0, 1.0), (4.0, 9.0)),
        ),
    )

    signatures = state_signatures_from_fitted_hmm(
        model,
        training_data_sha256=REFERENCE_DATA_SHA256,
    )

    np.testing.assert_allclose(
        tuple(item.location for item in signatures),
        ((1.0, 4.0), (2.0, -1.0)),
    )
    np.testing.assert_allclose(
        tuple(item.scale for item in signatures),
        ((np.sqrt(10.0), 4.0), (np.sqrt(8.0), 3.0)),
    )


def test_alignment_recovers_permuted_states_with_a_stable_digest() -> None:
    reference = (
        _signature(0, (-2.0, 0.0)),
        _signature(1, (2.0, 1.0)),
    )
    candidate = (
        _signature(
            0,
            (2.0, 1.0),
            training_data_sha256=CANDIDATE_DATA_SHA256,
        ),
        _signature(
            1,
            (-2.0, 0.0),
            training_data_sha256=CANDIDATE_DATA_SHA256,
        ),
    )

    result = align_states(_inputs(reference, candidate))

    assert result.permutation.candidate_to_reference == (1, 0)
    assert result.permutation.reference_to_candidate == (1, 0)
    assert result.total_cost == pytest.approx(0.0)
    assert len(result.alignment_sha256) == 64
    assert result.alignment_sha256 == align_states(
        _inputs(reference, candidate)
    ).alignment_sha256
    with pytest.raises(FrozenInstanceError):
        result.total_cost = 1.0  # type: ignore[misc]


def test_primary_cost_is_weighted_standardized_squared_distance() -> None:
    reference = (
        _signature(0, (0.0, 0.0), scale=(2.0, 4.0)),
        _signature(1, (10.0, 10.0)),
    )
    candidate = (
        _signature(
            0,
            (1.0, 4.0),
            scale=(4.0, 2.0),
            training_data_sha256=CANDIDATE_DATA_SHA256,
        ),
        _signature(
            1,
            (10.0, 10.0),
            training_data_sha256=CANDIDATE_DATA_SHA256,
        ),
    )
    config = StateAlignmentConfig(feature_weights=(1.0, 3.0))

    costs = build_state_alignment_cost_matrix(_inputs(reference, candidate), config)

    denominator = 0.5 * (2.0**2 + 4.0**2) + config.variance_floor
    expected = 0.25 * 1.0**2 / denominator + 0.75 * 4.0**2 / denominator
    assert costs[0][0] == pytest.approx(expected, rel=0.0, abs=1e-15)


def test_symmetric_gaussian_kl_is_zero_for_identical_signatures() -> None:
    reference = (
        _signature(0, (0.0, 0.0), scale=(1.0, 1.0)),
        _signature(1, (4.0, 2.0), scale=(2.0, 3.0)),
    )
    candidate = tuple(
        _signature(
            item.state_index,
            item.location,
            scale=item.scale,
            training_data_sha256=CANDIDATE_DATA_SHA256,
        )
        for item in reference
    )

    costs = build_state_alignment_cost_matrix(
        _inputs(reference, candidate),
        StateAlignmentConfig(metric=AlignmentMetric.SYMMETRIC_GAUSSIAN_KL),
    )

    assert costs[0][0] == pytest.approx(0.0)
    assert costs[1][1] == pytest.approx(0.0)
    assert costs[0][1] > 0.0


def test_lexicographic_ties_scale_without_permutation_enumeration() -> None:
    costs = np.zeros((12, 12), dtype=float)

    result = deterministic_hungarian_assignment(costs)

    assert result.candidate_to_reference == tuple(range(12))


def test_absolute_tie_band_does_not_replace_a_distinct_minimum() -> None:
    within_tolerance = deterministic_hungarian_assignment(
        ((0.5e-12, 0.0), (0.0, 0.5e-12))
    )
    outside_tolerance = deterministic_hungarian_assignment(
        ((0.5000001e-12, 0.0), (0.0, 0.5000001e-12))
    )

    assert within_tolerance.candidate_to_reference == (0, 1)
    assert outside_tolerance.candidate_to_reference == (1, 0)


def test_permutation_utilities_reorder_every_state_axis() -> None:
    permutation = StatePermutation((1, 0))

    assert reorder_candidate_values_to_reference(
        ("candidate-0", "candidate-1"),
        permutation,
    ) == ("candidate-1", "candidate-0")
    assert reorder_candidate_probabilities_to_reference(
        (0.8, 0.2),
        permutation,
    ) == pytest.approx((0.2, 0.8))
    canonical_transition = reorder_candidate_transition_matrix_to_reference(
        ((0.9, 0.1), (0.3, 0.7)),
        permutation,
    )
    np.testing.assert_allclose(canonical_transition, ((0.7, 0.3), (0.1, 0.9)))


def test_alignment_rejects_schema_and_training_provenance_drift() -> None:
    with pytest.raises(ValueError, match="share one training-data hash"):
        _inputs(
            (
                _signature(0, (0.0, 0.0)),
                _signature(
                    1,
                    (1.0, 1.0),
                    training_data_sha256="c" * 64,
                ),
            ),
            (
                _signature(
                    0,
                    (0.0, 0.0),
                    training_data_sha256=CANDIDATE_DATA_SHA256,
                ),
                _signature(
                    1,
                    (1.0, 1.0),
                    training_data_sha256=CANDIDATE_DATA_SHA256,
                ),
            ),
        )

    bad_schema = StateSignature(
        state_index=0,
        feature_names=SCHEMA.feature_names,
        feature_schema_sha256="d" * 64,
        location=(0.0, 0.0),
        scale=(1.0, 1.0),
        training_data_sha256=CANDIDATE_DATA_SHA256,
    )
    with pytest.raises(ValueError, match="authenticated schema"):
        _inputs(
            (_signature(0, (0.0, 0.0)), _signature(1, (1.0, 1.0))),
            (
                bad_schema,
                _signature(
                    1,
                    (1.0, 1.0),
                    training_data_sha256=CANDIDATE_DATA_SHA256,
                ),
            ),
        )


@pytest.mark.parametrize(
    "probabilities",
    ((0.8, np.nan), (0.8, 0.3), (-0.1, 1.1)),
)
def test_probability_reordering_rejects_invalid_simplexes(
    probabilities: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        reorder_candidate_probabilities_to_reference(
            probabilities,
            StatePermutation((0, 1)),
        )
