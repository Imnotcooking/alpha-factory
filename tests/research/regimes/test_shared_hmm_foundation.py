from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from oqp.contracts.regime_state import (
    OrderedFeatureSchema,
    ProbabilitySemantics,
    RegimeQualityFlag,
)
from oqp.research.ml.regimes.base import (
    DiagonalHMMConfig,
    HMMFamily,
    RegimeTrainer,
)
from oqp.research.ml.regimes.filtering import (
    CausalFilterSession,
    CausalFilteringError,
    FilterCheckpoint,
    FilterStartMode,
    filter_observation_sequence,
)
from oqp.research.ml.regimes.fitted import FittedDiagonalHMM
from oqp.research.ml.regimes.observations import ObservationBatch, ObservationSequence
from oqp.research.ml.regimes.serialization import (
    dump_filter_checkpoint_json,
    dump_fitted_hmm_json,
    load_filter_checkpoint_json,
    load_fitted_hmm_json,
    sha256_json,
)


BASE_TIME = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)


def _schema(
    schema_id: str = "m3_daily_v1", names: tuple[str, ...] = ("gk", "amihud")
) -> OrderedFeatureSchema:
    return OrderedFeatureSchema(schema_id=schema_id, feature_names=names)


def _model(family: HMMFamily = HMMFamily.GAUSSIAN) -> FittedDiagonalHMM:
    kwargs: dict[str, object] = {}
    if family is HMMFamily.GAUSSIAN_MIXTURE:
        means = np.array(
            [
                [[-0.5, -0.2], [0.4, 0.2]],
                [[1.7, 1.3], [2.6, 2.2]],
            ]
        )
        scales = np.array(
            [
                [[0.7, 1.1], [1.0, 0.8]],
                [[1.3, 0.9], [0.8, 1.4]],
            ]
        )
        kwargs["mixture_weights"] = np.array([[0.65, 0.35], [0.25, 0.75]])
    else:
        means = np.array([[0.0, -0.2], [2.1, 1.8]])
        scales = np.array([[0.8, 1.2], [1.4, 0.9]])
    if family is HMMFamily.STUDENT_T:
        kwargs["student_t_degrees_of_freedom"] = 8.0
    return FittedDiagonalHMM(
        model_id=f"synthetic-{family.value}",
        family=family,
        feature_schema=_schema(),
        initial_probabilities=np.array([0.6, 0.4]),
        transition_matrix=np.array([[0.91, 0.09], [0.18, 0.82]]),
        means=means,
        diagonal_scales=scales,
        state_ids=("quiet", "active"),
        training_run_id="synthetic-parity-fixture",
        **kwargs,
    )


def _sequence(
    values: np.ndarray,
    *,
    row_ids: tuple[str, ...] | None = None,
    sequence_id: str = "rb-segment-1",
    schema: OrderedFeatureSchema | None = None,
    entity_id: str = "SHFE.RB",
    observation_times: tuple[datetime, ...] | None = None,
) -> ObservationSequence:
    rows = row_ids or tuple(f"row-{index}" for index in range(len(values)))
    timestamps = observation_times or tuple(
        BASE_TIME + timedelta(days=index) for index in range(len(values))
    )
    return ObservationSequence(
        sequence_id=sequence_id,
        entity_id=entity_id,
        row_ids=rows,
        observation_times=timestamps,
        feature_schema=schema or _schema(),
        values=values,
    )


def test_observations_and_fitted_parameters_are_deeply_immutable() -> None:
    source = np.array([[0.1, 0.2], [0.3, 0.4]])
    sequence = _sequence(source)
    model = _model()
    source[0, 0] = 999.0

    assert sequence.values[0, 0] == pytest.approx(0.1)
    assert not sequence.values.flags.writeable
    assert not model.means.flags.writeable
    with pytest.raises(ValueError):
        sequence.values.setflags(write=True)
    with pytest.raises(ValueError):
        model.means[0, 0, 0] = 999.0


def test_observation_batch_keeps_explicit_sequence_boundaries() -> None:
    first = _sequence(np.array([[0.0, 0.0]]), sequence_id="rb-1")
    second = _sequence(np.array([[2.0, 2.0]]), sequence_id="cu-1")
    batch = ObservationBatch((first, second))

    assert batch.n_observations == 2
    assert batch.n_features == 2
    assert tuple(item.sequence_id for item in batch.sequences) == ("rb-1", "cu-1")


def test_schema_mismatch_is_rejected_before_emission_evaluation() -> None:
    sequence = _sequence(
        np.array([[0.0, 0.0]]),
        schema=_schema("reversed", ("amihud", "gk")),
    )
    with pytest.raises(ValueError, match="feature schema"):
        _model().log_emission_probabilities(sequence)
    with pytest.raises(TypeError, match="raw observations require"):
        _model().log_emission_probabilities(np.array([[0.0, 0.0]]))


@pytest.mark.parametrize(
    "family",
    [HMMFamily.GAUSSIAN, HMMFamily.GAUSSIAN_MIXTURE, HMMFamily.STUDENT_T],
)
def test_log_emissions_match_frozen_paper_01_engine(family: HMMFamily) -> None:
    """Characterize numerical parity without coupling the new model to Stage 7."""

    from engine.daily_regimes.hmm import (
        HMMFamily as FrozenFamily,
        _Parameters as FrozenParameters,
        _emission_log_probabilities as frozen_log_emissions,
    )

    model = _model(family)
    values = np.array([[-0.7, 0.1], [0.2, -0.4], [1.4, 1.0], [2.8, 2.5]])
    expected, _ = frozen_log_emissions(
        values,
        FrozenParameters(
            initial=model.initial_probabilities.copy(),
            transition=model.transition_matrix.copy(),
            mixture_weights=model.mixture_weights.copy(),
            means=model.means.copy(),
            scales=model.diagonal_scales.copy(),
        ),
        family=FrozenFamily(family.value),
        student_t_degrees_of_freedom=model.student_t_degrees_of_freedom,
    )

    actual = model.log_emission_probabilities(values, feature_schema=_schema())
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_causal_filter_matches_frozen_paper_01_log_space_recursion() -> None:
    from engine.daily_regimes.filtering import forward_filter_log_space

    model = _model()
    sequence = _sequence(np.array([[-0.4, -0.1], [0.2, 0.1], [1.0, 0.8], [2.4, 1.9]]))
    emissions = model.log_emission_probabilities(sequence)
    expected = forward_filter_log_space(
        model.initial_probabilities,
        model.transition_matrix,
        emissions,
    )
    actual = filter_observation_sequence(model, sequence)

    np.testing.assert_allclose(
        [step.observation_prior_probabilities for step in actual.steps],
        expected.observation_prior_probabilities,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        [step.filtered_probabilities for step in actual.steps],
        expected.filtered_probabilities,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        [step.one_step_probabilities for step in actual.steps],
        expected.one_step_predictive_probabilities,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        [step.log_predictive_density for step in actual.steps],
        expected.log_predictive_densities,
        atol=2e-14,
    )
    assert actual.log_likelihood == pytest.approx(expected.log_likelihood, abs=2e-14)


def test_checkpoint_continuation_equals_uninterrupted_filtering() -> None:
    model = _model(HMMFamily.GAUSSIAN_MIXTURE)
    values = np.array([[-0.4, -0.1], [0.2, 0.1], [1.0, 0.8], [2.4, 1.9], [1.8, 2.2]])
    one_shot = filter_observation_sequence(model, _sequence(values))

    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="rb-segment-1"
    )
    first_steps = tuple(
        session.update(
            values[index],
            feature_schema=_schema(),
            origin_row_id=f"row-{index}",
            observation_time=BASE_TIME + timedelta(days=index),
        )
        for index in range(2)
    )
    checkpoint = session.checkpoint()
    continuation = CausalFilterSession.continue_from_checkpoint(
        model,
        checkpoint,
        entity_id="SHFE.RB",
        sequence_id="rb-segment-1",
    )
    later_steps = tuple(
        continuation.update(
            values[index],
            feature_schema=_schema(),
            origin_row_id=f"row-{index}",
            observation_time=BASE_TIME + timedelta(days=index),
        )
        for index in range(2, len(values))
    )

    combined = (*first_steps, *later_steps)
    np.testing.assert_allclose(
        [step.filtered_probabilities for step in combined],
        [step.filtered_probabilities for step in one_shot.steps],
        atol=2e-14,
    )
    np.testing.assert_allclose(
        [step.log_predictive_density for step in combined],
        [step.log_predictive_density for step in one_shot.steps],
        atol=2e-14,
    )
    assert continuation.start_mode is FilterStartMode.CONTINUE


def test_checkpoint_cannot_cross_a_sequence_or_parameter_boundary() -> None:
    model = _model()
    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="rb-segment-1"
    )
    session.update(
        [0.0, 0.0],
        feature_schema=_schema(),
        origin_row_id="row-0",
        observation_time=BASE_TIME,
    )
    checkpoint = session.checkpoint()

    with pytest.raises(CausalFilteringError, match="sequence boundary"):
        CausalFilterSession.continue_from_checkpoint(
            model,
            checkpoint,
            entity_id="SHFE.RB",
            sequence_id="rb-segment-2",
        )

    changed = FittedDiagonalHMM(
        model_id=model.model_id,
        family=model.family,
        feature_schema=model.feature_schema,
        initial_probabilities=model.initial_probabilities,
        transition_matrix=np.array([[0.8, 0.2], [0.1, 0.9]]),
        means=model.means,
        diagonal_scales=model.diagonal_scales,
        state_ids=model.state_ids,
        training_run_id=model.training_run_id,
    )
    with pytest.raises(CausalFilteringError, match="parameter digest"):
        CausalFilterSession.continue_from_checkpoint(
            changed,
            checkpoint,
            entity_id="SHFE.RB",
            sequence_id="rb-segment-1",
        )


def test_fitted_model_json_round_trip_and_tamper_detection(tmp_path) -> None:
    model = _model(HMMFamily.STUDENT_T)
    path = dump_fitted_hmm_json(model, tmp_path / "model.json")
    restored = load_fitted_hmm_json(
        path,
        expected_model_id=model.model_id,
        expected_parameter_sha256=model.parameter_sha256,
    )

    assert restored.state_dict() == model.state_dict()
    assert restored.identity == model.identity

    tampered = model.state_dict()
    tampered["means"][0][0][0] += 0.25
    with pytest.raises(ValueError, match="does not authenticate"):
        FittedDiagonalHMM.from_state_dict(tampered)


def test_checkpoint_json_round_trip_and_tamper_detection(tmp_path) -> None:
    model = _model()
    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="rb-segment-1"
    )
    session.update(
        [0.0, 0.0],
        feature_schema=_schema(),
        origin_row_id="row-0",
        observation_time=BASE_TIME,
    )
    checkpoint = session.checkpoint()
    path = dump_filter_checkpoint_json(checkpoint, tmp_path / "checkpoint.json")
    restored = load_filter_checkpoint_json(
        path,
        expected_model_id=model.model_id,
        expected_parameter_sha256=model.parameter_sha256,
        expected_entity_id="SHFE.RB",
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )

    assert restored == checkpoint
    assert restored.semantics is ProbabilitySemantics.ONE_STEP_PREDICTED

    tampered = checkpoint.state_dict()
    tampered["probabilities"][0] += 0.01
    tampered["probabilities"][1] -= 0.01
    with pytest.raises(ValueError, match="does not authenticate"):
        FilterCheckpoint.from_state_dict(
            tampered,
            expected_model_id=model.model_id,
            expected_parameter_sha256=model.parameter_sha256,
            expected_entity_id="SHFE.RB",
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )

    forged = checkpoint.state_dict()
    forged["probabilities"] = [0.99, 0.01]
    forged["checkpoint_sha256"] = sha256_json(
        {key: value for key, value in forged.items() if key != "checkpoint_sha256"}
    )
    with pytest.raises(ValueError, match="independently trusted digest"):
        FilterCheckpoint.from_state_dict(
            forged,
            expected_model_id=model.model_id,
            expected_parameter_sha256=model.parameter_sha256,
            expected_entity_id="SHFE.RB",
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )


def test_operational_inference_uses_shared_contract_and_flags_reset() -> None:
    model = _model()
    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="rb-segment-1"
    )
    observed = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)
    inferred = observed + timedelta(minutes=1)
    predicted = observed + timedelta(days=1)

    inference = session.infer(
        [0.1, -0.1],
        feature_schema=_schema(),
        origin_row_id="2026-07-20",
        observation_time=observed,
        inference_time=inferred,
        prediction_time=predicted,
    )

    assert inference.model.artifact_sha256 == model.parameter_sha256
    assert inference.model.feature_schema_sha256 == model.feature_schema.schema_sha256
    assert inference.filtered_probabilities == inference.probabilities_for(
        ProbabilitySemantics.FILTERED
    )
    assert RegimeQualityFlag.STATE_RESET in inference.quality_flags
    assert inference.log_predictive_density is not None
    assert np.isfinite(inference.log_predictive_density)


def test_operational_filter_enforces_schema_entity_and_monotonic_time() -> None:
    model = _model()
    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="segment-1"
    )
    session.update(
        [0.0, 0.0],
        feature_schema=_schema(),
        origin_row_id="row-1",
        observation_time=BASE_TIME,
    )

    with pytest.raises(CausalFilteringError, match="advance strictly"):
        session.update(
            [0.0, 0.0],
            feature_schema=_schema(),
            origin_row_id="row-0",
            observation_time=BASE_TIME - timedelta(days=1),
        )
    with pytest.raises(CausalFilteringError, match="feature schema"):
        session.update(
            [0.0, 0.0],
            feature_schema=_schema("reordered", ("amihud", "gk")),
            origin_row_id="row-2",
            observation_time=BASE_TIME + timedelta(days=1),
        )

    checkpoint = session.checkpoint()
    with pytest.raises(CausalFilteringError, match="entity boundary"):
        CausalFilterSession.continue_from_checkpoint(
            model,
            checkpoint,
            entity_id="DCE.M",
            sequence_id="segment-1",
        )


def test_failed_operational_inference_rolls_back_session_state() -> None:
    model = _model()
    session = CausalFilterSession.reset(
        model, entity_id="SHFE.RB", sequence_id="segment-1"
    )
    with pytest.raises(ValueError, match="strictly after"):
        session.infer(
            [0.0, 0.0],
            feature_schema=_schema(),
            origin_row_id="row-0",
            observation_time=BASE_TIME,
            inference_time=BASE_TIME + timedelta(minutes=1),
            prediction_time=BASE_TIME + timedelta(minutes=1),
        )

    recovered = session.infer(
        [0.0, 0.0],
        feature_schema=_schema(),
        origin_row_id="row-0",
        observation_time=BASE_TIME,
        inference_time=BASE_TIME + timedelta(minutes=1),
        prediction_time=BASE_TIME + timedelta(days=1),
    )
    assert RegimeQualityFlag.STATE_RESET in recovered.quality_flags


def test_model_id_is_bound_to_artifact_hash_and_trusted_loader(tmp_path) -> None:
    model = _model()
    state = model.state_dict()
    state["model_id"] = "approved-looking-name"
    with pytest.raises(ValueError, match="does not authenticate"):
        FittedDiagonalHMM.from_state_dict(state)

    path = dump_fitted_hmm_json(model, tmp_path / "model.json")
    with pytest.raises(ValueError, match="expected_model_id"):
        load_fitted_hmm_json(
            path,
            expected_model_id="wrong-model",
            expected_parameter_sha256=model.parameter_sha256,
        )


def test_json_artifact_writes_require_explicit_overwrite(tmp_path) -> None:
    model = _model()
    path = dump_fitted_hmm_json(model, tmp_path / "model.json")
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        dump_fitted_hmm_json(model, path)
    assert path.read_bytes() == original

    dump_fitted_hmm_json(model, path, overwrite=True)
    restored = load_fitted_hmm_json(
        path,
        expected_model_id=model.model_id,
        expected_parameter_sha256=model.parameter_sha256,
    )
    assert restored.parameter_sha256 == model.parameter_sha256


def test_configuration_and_trainer_protocol_are_backend_independent() -> None:
    config = DiagonalHMMConfig(
        family=HMMFamily.STUDENT_T,
        n_states=3,
        student_t_degrees_of_freedom=8.0,
    )

    class StructuralTrainer:
        def fit(self, batch, config, *, model_id, training_run_id=None):
            raise NotImplementedError

    assert config.n_states == 3
    assert isinstance(StructuralTrainer(), RegimeTrainer)
    with pytest.raises(ValueError, match="n_mixtures"):
        DiagonalHMMConfig(
            family=HMMFamily.GAUSSIAN,
            n_states=2,
            n_mixtures=2,
        )
