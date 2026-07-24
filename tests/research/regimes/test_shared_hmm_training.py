from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from oqp.contracts.regime_state import OrderedFeatureSchema
from oqp.research.ml.regimes.base import (
    DiagonalHMMConfig,
    HMMFamily,
    RegimeTrainer,
)
from oqp.research.ml.regimes.fitted import FittedDiagonalHMM
from oqp.research.ml.regimes.observations import ObservationBatch, ObservationSequence
from oqp.research.ml.regimes.trainer import (
    DeterministicDiagonalHMMTrainer,
    DiagonalHMMTrainingControls,
    HMMTrainingError,
    HMMTrainingFailure,
    _expectation,
    _fit_one_restart,
    _initial_parameters,
    _prepare_batch,
    hash_observation_batch,
)


BASE_TIME = datetime(2025, 1, 1, 15, 0, tzinfo=timezone.utc)
PREPROCESSING_SHA256 = "a" * 64


def _schema() -> OrderedFeatureSchema:
    return OrderedFeatureSchema(
        schema_id="synthetic-regime-input-v1",
        feature_names=("gk_volatility", "amihud_illiquidity"),
    )


def _sequence(
    sequence_id: str,
    entity_id: str,
    values: np.ndarray,
) -> ObservationSequence:
    return ObservationSequence(
        sequence_id=sequence_id,
        entity_id=entity_id,
        row_ids=tuple(f"{sequence_id}-row-{index}" for index in range(len(values))),
        observation_times=tuple(
            BASE_TIME + timedelta(days=index) for index in range(len(values))
        ),
        feature_schema=_schema(),
        values=values,
    )


def _batch() -> ObservationBatch:
    rng = np.random.default_rng(20260723)
    first = np.concatenate(
        (
            rng.normal((-1.8, -1.0), (0.30, 0.25), size=(35, 2)),
            rng.normal((1.7, 1.2), (0.40, 0.35), size=(35, 2)),
        )
    )
    second = np.concatenate(
        (
            rng.normal((1.8, 1.1), (0.35, 0.40), size=(30, 2)),
            rng.normal((-1.7, -1.1), (0.25, 0.30), size=(30, 2)),
        )
    )
    return ObservationBatch(
        (
            _sequence("rb-segment-1", "SHFE.RB", first),
            _sequence("cu-segment-1", "SHFE.CU", second),
        )
    )


def _config(family: HMMFamily) -> DiagonalHMMConfig:
    return DiagonalHMMConfig(
        family=family,
        n_states=2,
        n_mixtures=2 if family is HMMFamily.GAUSSIAN_MIXTURE else 1,
        student_t_degrees_of_freedom=(8.0 if family is HMMFamily.STUDENT_T else None),
    )


def _controls(**changes: object) -> DiagonalHMMTrainingControls:
    values: dict[str, object] = {
        "n_restarts": 2,
        "max_iterations": 150,
        "tolerance_per_observation": 1e-6,
        "minimum_state_occupancy": 0.01,
        "random_seed": 147,
    }
    values.update(changes)
    return DiagonalHMMTrainingControls(**values)


def test_trainer_implements_shared_protocol_and_is_exactly_deterministic() -> None:
    trainer = DeterministicDiagonalHMMTrainer(_controls(n_restarts=3))
    batch = _batch()
    config = _config(HMMFamily.GAUSSIAN)

    first = trainer.fit_with_diagnostics(
        batch,
        config,
        model_id="daily-risk-regimes-v1",
        training_run_id="synthetic-fit-1",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )
    second = trainer.fit_with_diagnostics(
        batch,
        config,
        model_id="daily-risk-regimes-v1",
        training_run_id="synthetic-fit-1",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )

    assert isinstance(trainer, RegimeTrainer)
    assert first.model.state_dict() == second.model.state_dict()
    assert first.restarts == second.restarts
    assert first.selected_restart_index == second.selected_restart_index
    assert first.training_data_sha256 == second.training_data_sha256
    assert first.training_controls_sha256 == second.training_controls_sha256
    assert first.model.parameter_sha256 == second.model.parameter_sha256
    assert first.model.preprocessing_artifact_sha256 == PREPROCESSING_SHA256
    assert len(first.restarts) == 3
    assert tuple(item.restart_index for item in first.restarts) == (0, 1, 2)
    assert first.restarts[first.selected_restart_index].accepted


@pytest.mark.parametrize(
    "family",
    [HMMFamily.GAUSSIAN, HMMFamily.GAUSSIAN_MIXTURE, HMMFamily.STUDENT_T],
)
def test_all_supported_families_fit_the_shared_immutable_geometry(
    family: HMMFamily,
) -> None:
    result = DeterministicDiagonalHMMTrainer(_controls()).fit_with_diagnostics(
        _batch(),
        _config(family),
        model_id=f"synthetic-{family.value}",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )
    model = result.model

    assert model.family is family
    assert model.means.shape == (
        2,
        2 if family is HMMFamily.GAUSSIAN_MIXTURE else 1,
        2,
    )
    assert not model.means.flags.writeable
    assert np.isfinite(model.log_emission_probabilities(_batch().sequences[0])).all()
    if family is HMMFamily.STUDENT_T:
        assert model.student_t_degrees_of_freedom == 8.0
    else:
        assert model.student_t_degrees_of_freedom is None


def test_sequence_boundaries_reset_initial_state_and_create_no_transition() -> None:
    first = _sequence("a", "A", np.array([[-2.0, -1.0]]))
    second = _sequence("b", "B", np.array([[2.0, 1.0]]))
    separated = _prepare_batch(ObservationBatch((first, second)))
    config = _config(HMMFamily.GAUSSIAN)
    controls = _controls(n_restarts=1)
    parameters = _initial_parameters(
        separated.all_values,
        config=config,
        controls=controls,
        seed=17,
    )

    separated_expectation = _expectation(separated, parameters, config=config)
    joined = _prepare_batch(
        ObservationBatch(
            (_sequence("joined", "A", np.array([[-2.0, -1.0], [2.0, 1.0]])),)
        )
    )
    joined_expectation = _expectation(joined, parameters, config=config)

    np.testing.assert_array_equal(separated_expectation.transition_counts, 0.0)
    assert joined_expectation.transition_counts.sum() == pytest.approx(1.0)
    assert separated_expectation.start_counts.sum() == pytest.approx(2.0)
    assert joined_expectation.start_counts.sum() == pytest.approx(1.0)


def test_batch_order_is_semantically_irrelevant_but_sequence_membership_is_hashed() -> (
    None
):
    batch = _batch()
    reversed_batch = ObservationBatch(tuple(reversed(batch.sequences)))

    assert hash_observation_batch(batch) == hash_observation_batch(reversed_batch)
    first = DeterministicDiagonalHMMTrainer(_controls()).fit(
        batch,
        _config(HMMFamily.GAUSSIAN),
        model_id="order-invariant",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )
    second = DeterministicDiagonalHMMTrainer(_controls()).fit(
        reversed_batch,
        _config(HMMFamily.GAUSSIAN),
        model_id="order-invariant",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )
    assert first.parameter_sha256 == second.parameter_sha256


def test_preprocessing_digest_is_validated_serialized_and_parameter_bound() -> None:
    trainer = DeterministicDiagonalHMMTrainer(_controls())
    batch = _batch()
    config = _config(HMMFamily.GAUSSIAN)
    first = trainer.fit(
        batch,
        config,
        model_id="lineage-bound",
        preprocessing_artifact_sha256="a" * 64,
    )
    second = trainer.fit(
        batch,
        config,
        model_id="lineage-bound",
        preprocessing_artifact_sha256="b" * 64,
    )

    assert first.parameter_sha256 != second.parameter_sha256
    assert first.state_dict()["preprocessing_artifact_sha256"] == "a" * 64
    restored = FittedDiagonalHMM.from_state_dict(first.state_dict())
    assert restored.preprocessing_artifact_sha256 == "a" * 64
    assert restored.parameter_sha256 == first.parameter_sha256
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        trainer.fit(
            batch,
            config,
            model_id="invalid-lineage",
            preprocessing_artifact_sha256="A" * 64,
        )


def test_trainer_does_not_touch_numpy_global_random_state() -> None:
    state_before = np.random.get_state()
    DeterministicDiagonalHMMTrainer(_controls()).fit(
        _batch(),
        _config(HMMFamily.GAUSSIAN),
        model_id="private-rng",
        preprocessing_artifact_sha256=PREPROCESSING_SHA256,
    )
    state_after = np.random.get_state()

    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_failed_fit_exposes_every_restart_without_returning_a_model() -> None:
    trainer = DeterministicDiagonalHMMTrainer(_controls(n_restarts=3, max_iterations=1))

    with pytest.raises(HMMTrainingError) as captured:
        trainer.fit(
            _batch(),
            _config(HMMFamily.GAUSSIAN),
            model_id="intentionally-nonconvergent",
            preprocessing_artifact_sha256=PREPROCESSING_SHA256,
        )

    assert len(captured.value.restarts) == 3
    assert all(not item.accepted for item in captured.value.restarts)
    assert all(
        HMMTrainingFailure.NON_CONVERGENCE in item.failure_codes
        for item in captured.value.restarts
    )


@pytest.mark.parametrize(
    "family",
    [HMMFamily.GAUSSIAN, HMMFamily.GAUSSIAN_MIXTURE, HMMFamily.STUDENT_T],
)
def test_one_restart_matches_frozen_paper01_numerical_engine(
    family: HMMFamily,
) -> None:
    """Characterize the port without importing frozen code in production."""

    from engine.daily_regimes.hmm import (
        CovarianceType as FrozenCovarianceType,
        HMMConfig as FrozenConfig,
        HMMFamily as FrozenFamily,
        HMMSequence as FrozenSequence,
        HMMTrainingBatch as FrozenBatch,
        _fit_one_restart as frozen_fit_one_restart,
        _prepare_batch as frozen_prepare_batch,
    )

    batch = _batch()
    config = _config(family)
    controls = _controls(n_restarts=1)
    seed = 2048
    actual = _fit_one_restart(
        _prepare_batch(batch),
        config=config,
        controls=controls,
        seed=seed,
    )
    frozen_sequences = tuple(
        FrozenSequence(
            sequence_id=sequence.sequence_id,
            product_id=sequence.entity_id,
            row_ids=sequence.row_ids,
            input_columns=sequence.feature_schema.feature_names,
            values=tuple(
                tuple(float(value) for value in row) for row in sequence.values
            ),
        )
        for sequence in batch.sequences
    )
    frozen_config = FrozenConfig(
        family=FrozenFamily(family.value),
        n_states=config.n_states,
        n_mixtures=config.n_mixtures,
        covariance_type=FrozenCovarianceType.DIAGONAL,
        n_restarts=1,
        max_iterations=controls.max_iterations,
        tolerance=controls.tolerance_per_observation,
        covariance_floor=config.covariance_floor,
        minimum_state_occupancy=controls.minimum_state_occupancy,
        random_seed=controls.random_seed,
        student_t_degrees_of_freedom=(config.student_t_degrees_of_freedom or 8.0),
        initial_self_transition_probability=(
            controls.initial_self_transition_probability
        ),
        probability_pseudocount=controls.probability_pseudocount,
        probability_floor=config.probability_floor,
    )
    expected = frozen_fit_one_restart(
        frozen_prepare_batch(
            FrozenBatch(
                fold_id="synthetic",
                feature_set_id="m2",
                preprocessing_artifact_hash=PREPROCESSING_SHA256,
                sequences=frozen_sequences,
            )
        ),
        config=frozen_config,
        seed=seed,
    )

    assert actual.converged == expected.converged
    assert actual.iterations == expected.iterations
    assert actual.expectation.log_likelihood == pytest.approx(
        expected.expectation.log_likelihood, abs=2e-10
    )
    np.testing.assert_allclose(
        actual.parameters.initial, expected.parameters.initial, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        actual.parameters.transition,
        expected.parameters.transition,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        actual.parameters.means, expected.parameters.means, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        actual.parameters.scales, expected.parameters.scales, rtol=1e-10, atol=1e-10
    )
