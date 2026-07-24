from __future__ import annotations

from datetime import datetime, timedelta, timezone
from importlib import import_module

import numpy as np
import pytest

from oqp.contracts.regime_state import OrderedFeatureSchema
from oqp.research.ml.regimes import GMMHMM, GaussianHMM, StudentTHMM
from oqp.research.ml.regimes.base import DiagonalHMMConfig, HMMFamily
from oqp.research.ml.regimes.observations import ObservationBatch, ObservationSequence
from oqp.research.ml.regimes.trainer import (
    DeterministicDiagonalHMMTrainer,
    DiagonalHMMTrainingControls,
)


def _batch() -> ObservationBatch:
    rng = np.random.default_rng(20260723)
    values = np.concatenate(
        (
            rng.normal((-1.5, -0.8), (0.25, 0.30), size=(35, 2)),
            rng.normal((1.6, 1.0), (0.30, 0.25), size=(35, 2)),
        )
    )
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    schema = OrderedFeatureSchema(
        schema_id="named-hmm-fixture-v1",
        feature_names=("gk_volatility", "amihud_illiquidity"),
    )
    sequence = ObservationSequence(
        sequence_id="fixture-1",
        entity_id="SHFE.RB",
        row_ids=tuple(f"row-{index}" for index in range(len(values))),
        observation_times=tuple(
            base_time + timedelta(days=index) for index in range(len(values))
        ),
        feature_schema=schema,
        values=values,
    )
    return ObservationBatch((sequence,))


def _controls() -> DiagonalHMMTrainingControls:
    return DiagonalHMMTrainingControls(
        n_restarts=1,
        max_iterations=150,
        tolerance_per_observation=1e-6,
        minimum_state_occupancy=0.01,
        random_seed=91,
    )


@pytest.mark.parametrize(
    ("estimator", "family", "mixtures", "degrees"),
    (
        (GaussianHMM(2, controls=_controls()), HMMFamily.GAUSSIAN, 1, None),
        (GMMHMM(2, controls=_controls()), HMMFamily.GAUSSIAN_MIXTURE, 2, None),
        (StudentTHMM(2, controls=_controls()), HMMFamily.STUDENT_T, 1, 8.0),
    ),
)
def test_named_estimators_fit_their_exact_family_via_shared_backend(
    estimator: GaussianHMM | GMMHMM | StudentTHMM,
    family: HMMFamily,
    mixtures: int,
    degrees: float | None,
) -> None:
    config = estimator.config
    assert isinstance(config, DiagonalHMMConfig)
    assert config.family is family
    assert config.n_states == 2
    assert config.n_mixtures == mixtures
    assert config.student_t_degrees_of_freedom == degrees

    result = estimator.fit_with_diagnostics(
        _batch(),
        model_id=f"named-{family.value}",
        training_run_id="named-api-test",
        preprocessing_artifact_sha256="a" * 64,
    )

    assert result.model.family is family
    assert result.model.n_states == 2
    assert result.model.n_mixtures == mixtures
    assert result.model.student_t_degrees_of_freedom == degrees
    assert result.model.preprocessing_artifact_sha256 == "a" * 64
    assert len(result.restarts) == 1


def test_named_fit_returns_only_the_selected_model() -> None:
    estimator = GaussianHMM(2, controls=_controls())
    model = estimator.fit(
        _batch(),
        model_id="named-gaussian-fit",
        preprocessing_artifact_sha256="b" * 64,
    )

    assert model.family is HMMFamily.GAUSSIAN
    assert model.preprocessing_artifact_sha256 == "b" * 64


@pytest.mark.parametrize(
    "estimator",
    (
        GaussianHMM(2, controls=_controls()),
        GMMHMM(2, controls=_controls()),
        StudentTHMM(2, controls=_controls()),
    ),
)
def test_named_estimator_is_numerically_identical_to_the_shared_backend(
    estimator: GaussianHMM | GMMHMM | StudentTHMM,
) -> None:
    batch = _batch()
    model_id = f"parity-{estimator.config.family.value}"
    named = estimator.fit_with_diagnostics(
        batch,
        model_id=model_id,
        training_run_id="named-backend-parity",
        preprocessing_artifact_sha256="c" * 64,
    )
    direct = DeterministicDiagonalHMMTrainer(estimator.controls).fit_with_diagnostics(
        batch,
        estimator.config,
        model_id=model_id,
        training_run_id="named-backend-parity",
        preprocessing_artifact_sha256="c" * 64,
    )

    assert named.model.state_dict() == direct.model.state_dict()
    assert named.restarts == direct.restarts
    assert named.training_data_sha256 == direct.training_data_sha256
    assert named.training_controls_sha256 == direct.training_controls_sha256


def test_named_estimators_validate_family_specific_geometry_at_construction() -> None:
    with pytest.raises(ValueError, match="at least two"):
        GaussianHMM(1)
    with pytest.raises(ValueError, match="at least two mixtures"):
        GMMHMM(2, n_mixtures=1)
    with pytest.raises(ValueError, match="above two"):
        StudentTHMM(2, degrees_of_freedom=2.0)

    assert GaussianHMM.__module__ == "oqp.research.ml.regimes.gaussian_hmm"
    assert GMMHMM.__module__ == "oqp.research.ml.regimes.gmm_hmm"
    assert StudentTHMM.__module__ == "oqp.research.ml.regimes.student_t_hmm"


def test_private_trainer_parity_hooks_and_legacy_market_hmm_are_canonical() -> None:
    new_trainer = import_module("oqp.research.ml.regimes.trainer")
    assert callable(new_trainer._fit_one_restart)
    assert callable(new_trainer._expectation)

    new_hmm = import_module("oqp.research.ml.regimes.hmm")
    assert (
        new_hmm.MarketHMM.__module__
        == "oqp.research.ml.regimes.legacy.hmmlearn_models"
    )
    assert (
        new_hmm.MarketGMMHMM.__module__
        == "oqp.research.ml.regimes.legacy.hmmlearn_models"
    )


def test_named_estimators_are_exported_from_the_canonical_package() -> None:
    canonical = import_module("oqp.research.ml.regimes")

    assert canonical.GaussianHMM is GaussianHMM
    assert canonical.GMMHMM is GMMHMM
    assert canonical.StudentTHMM is StudentTHMM
