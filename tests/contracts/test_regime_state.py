from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from oqp.contracts.regime_state import (
    ModelIdentity,
    OrderedFeatureSchema,
    ProbabilitySemantics,
    RegimeInference,
    RegimeQualityFlag,
)


OBSERVED = datetime(2026, 7, 10, 15, 0, tzinfo=timezone(timedelta(hours=8)))
INFERRED = datetime(2026, 7, 10, 15, 1, tzinfo=timezone(timedelta(hours=8)))
PREDICTED = datetime(2026, 7, 13, 15, 0, tzinfo=timezone(timedelta(hours=8)))


def _schema() -> OrderedFeatureSchema:
    return OrderedFeatureSchema(
        schema_id="cn-futures-daily-m3",
        schema_version=1,
        feature_names=("gk_gap", "amihud", "ker_20"),
    )


def _model(schema: OrderedFeatureSchema | None = None) -> ModelIdentity:
    bound_schema = schema or _schema()
    return ModelIdentity(
        model_id="gaussian-hmm-m3-k2-fold-4",
        model_family="gaussian_hmm",
        model_version="1.0.0",
        artifact_sha256="a" * 64,
        feature_schema_sha256=bound_schema.schema_sha256,
        training_run_id="paper01-stage12-fold4",
    )


def _inference(**overrides: object) -> RegimeInference:
    schema = overrides.pop("feature_schema", _schema())
    values: dict[str, object] = {
        "entity_id": "CZCE.CF",
        "sequence_id": "CZCE.CF:continuous:2026",
        "observation_time": OBSERVED,
        "inference_time": INFERRED,
        "prediction_time": PREDICTED,
        "model": _model(schema),
        "feature_schema": schema,
        "state_ids": ("state_0", "state_1"),
        "filtered_probabilities": (0.2, 0.8),
        "one_step_probabilities": (0.35, 0.65),
        "dominant_state": "state_1",
        "semantic_label": "high-volatility",
        "log_predictive_density": -4.382026634673881,
        "quality_flags": (RegimeQualityFlag.STATE_RESET,),
    }
    values.update(overrides)
    return RegimeInference(**values)


def test_contracts_are_frozen_slotted_and_json_round_trip() -> None:
    original = _inference()
    encoded = json.loads(json.dumps(original.state_dict(), allow_nan=False))
    restored = RegimeInference.from_state_dict(
        encoded,
        expected_model_id=original.model.model_id,
        expected_artifact_sha256=original.model.artifact_sha256,
        expected_feature_schema_sha256=original.feature_schema.schema_sha256,
    )

    assert restored == original
    assert restored.state_dict() == encoded
    assert restored.probabilities_for(ProbabilitySemantics.FILTERED) == (0.2, 0.8)
    assert restored.probabilities_for(ProbabilitySemantics.ONE_STEP_PREDICTED) == (
        0.35,
        0.65,
    )
    assert not hasattr(original, "__dict__")
    with pytest.raises(FrozenInstanceError):
        original.entity_id = "DCE.M"  # type: ignore[misc]


def test_probability_semantics_include_smoothed_but_inference_rejects_it() -> None:
    assert {item.value for item in ProbabilitySemantics} == {
        "filtered",
        "one_step_predicted",
        "smoothed",
    }
    with pytest.raises(ValueError, match="never contains smoothed"):
        _inference().probabilities_for(ProbabilitySemantics.SMOOTHED)


def test_ordered_schema_hash_authenticates_names_and_order() -> None:
    first = _schema()
    reordered = OrderedFeatureSchema(
        schema_id=first.schema_id,
        schema_version=first.schema_version,
        feature_names=tuple(reversed(first.feature_names)),
    )
    assert first.schema_sha256 != reordered.schema_sha256

    tampered = first.state_dict()
    tampered["feature_names"] = ["gk_gap", "ker_20", "amihud"]
    with pytest.raises(ValueError, match="does not authenticate"):
        OrderedFeatureSchema.from_state_dict(tampered)

    with pytest.raises(ValueError, match="unique"):
        OrderedFeatureSchema(schema_id="bad", feature_names=("gk", "gk"))


@pytest.mark.parametrize(
    "digest",
    ["a" * 63, "A" * 64, "z" * 64, 123],
)
def test_model_identity_rejects_noncanonical_hashes(digest: object) -> None:
    schema = _schema()
    with pytest.raises(ValueError, match="SHA-256"):
        ModelIdentity(
            model_id="model",
            model_family="hmm",
            model_version="1",
            artifact_sha256=digest,  # type: ignore[arg-type]
            feature_schema_sha256=schema.schema_sha256,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("filtered_probabilities", (0.2, 0.7)),
        ("filtered_probabilities", (-0.1, 1.1)),
        ("filtered_probabilities", (float("nan"), float("nan"))),
        ("one_step_probabilities", (1.0,)),
    ],
)
def test_inference_rejects_invalid_probability_simplexes(
    field_name: str, value: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="simplex|finite|length"):
        _inference(**{field_name: value})


def test_inference_rejects_naive_or_noncausal_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _inference(observation_time=OBSERVED.replace(tzinfo=None))
    with pytest.raises(ValueError, match="cannot precede"):
        _inference(inference_time=OBSERVED - timedelta(seconds=1))
    with pytest.raises(ValueError, match="strictly after"):
        _inference(prediction_time=INFERRED)


def test_inference_rejects_model_schema_mismatch() -> None:
    schema = _schema()
    wrong_model = ModelIdentity(
        model_id="wrong-schema-model",
        model_family="gaussian_hmm",
        model_version="1",
        artifact_sha256="b" * 64,
        feature_schema_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="does not match feature_schema"):
        _inference(feature_schema=schema, model=wrong_model)


def test_inference_rejects_invalid_dominant_state_and_quality_flag() -> None:
    with pytest.raises(ValueError, match="maximum filtered"):
        _inference(dominant_state="state_0")
    with pytest.raises(ValueError, match="unknown regime quality flag"):
        _inference(quality_flags=("mystery",))


def test_deserialization_rejects_schema_and_identity_drift() -> None:
    original = _inference()
    state = json.loads(json.dumps(original.state_dict()))
    state["unknown"] = True
    with pytest.raises(ValueError, match="frozen schema"):
        RegimeInference.from_state_dict(state)

    clean = json.loads(json.dumps(original.state_dict()))
    with pytest.raises(ValueError, match="expected_model_id"):
        RegimeInference.from_state_dict(clean, expected_model_id="another-model")

    clean["probabilities"]["smoothed"] = [0.1, 0.9]
    with pytest.raises(ValueError, match="frozen schema"):
        RegimeInference.from_state_dict(clean)


def test_serialized_timestamps_must_be_canonical_and_aware() -> None:
    state = _inference().state_dict()
    state["observation_time"] = "2026-07-10T07:00:00Z"
    with pytest.raises(ValueError, match="canonical"):
        RegimeInference.from_state_dict(state)

    state = _inference().state_dict()
    state["observation_time"] = "2026-07-10T15:00:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        RegimeInference.from_state_dict(state)
