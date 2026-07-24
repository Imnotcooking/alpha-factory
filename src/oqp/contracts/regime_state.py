"""Dependency-light contracts for regime-model identity and causal inference.

The contracts in this module deliberately depend only on the Python standard
library.  Research estimators, dashboards, and operational inference services
can therefore exchange authenticated model metadata and state probabilities
without importing a numerical or machine-learning runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import fsum, isfinite
from typing import Any, ClassVar


_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_SIMPLEX_TOLERANCE = 1e-9


class ProbabilitySemantics(str, Enum):
    """The information set used to calculate a state-probability vector."""

    FILTERED = "filtered"
    ONE_STEP_PREDICTED = "one_step_predicted"
    SMOOTHED = "smoothed"


class RegimeQualityFlag(str, Enum):
    """Non-fatal conditions attached to an otherwise valid inference."""

    WARMUP = "warmup"
    STATE_RESET = "state_reset"
    MISSING_INPUT = "missing_input"
    STALE_INPUT = "stale_input"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    NUMERICAL_FALLBACK = "numerical_fallback"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable identity of a fitted model artifact.

    ``artifact_sha256`` authenticates the fitted parameters or model bundle.
    ``feature_schema_sha256`` binds the artifact to the exact ordered input
    schema it was trained to consume.
    """

    CONTRACT_VERSION: ClassVar[int] = 1

    model_id: str
    model_family: str
    model_version: str
    artifact_sha256: str
    feature_schema_sha256: str
    training_run_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, "model_id")
        _require_identifier(self.model_family, "model_family")
        _require_identifier(self.model_version, "model_version")
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_sha256(self.feature_schema_sha256, "feature_schema_sha256")
        if self.training_run_id is not None:
            _require_identifier(self.training_run_id, "training_run_id")

    def state_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-safe representation."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "artifact_sha256": self.artifact_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "training_run_id": self.training_run_id,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_model_id: str | None = None,
        expected_artifact_sha256: str | None = None,
    ) -> ModelIdentity:
        """Restore an identity, rejecting schema or expected-identity drift."""

        payload = _require_mapping(state, "model identity")
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "model_id",
                "model_family",
                "model_version",
                "artifact_sha256",
                "feature_schema_sha256",
                "training_run_id",
            },
            "model identity",
        )
        _require_contract_version(payload["contract_version"], cls.CONTRACT_VERSION)
        identity = cls(
            model_id=payload["model_id"],
            model_family=payload["model_family"],
            model_version=payload["model_version"],
            artifact_sha256=payload["artifact_sha256"],
            feature_schema_sha256=payload["feature_schema_sha256"],
            training_run_id=payload["training_run_id"],
        )
        if expected_model_id is not None and identity.model_id != expected_model_id:
            raise ValueError("model identity does not match expected_model_id")
        if (
            expected_artifact_sha256 is not None
            and identity.artifact_sha256 != expected_artifact_sha256
        ):
            raise ValueError("model identity does not match expected_artifact_sha256")
        return identity


@dataclass(frozen=True, slots=True)
class OrderedFeatureSchema:
    """Versioned, ordered feature names with a self-verifying digest."""

    CONTRACT_VERSION: ClassVar[int] = 1

    schema_id: str
    feature_names: tuple[str, ...]
    schema_version: int = 1
    schema_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.schema_id, "schema_id")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")

        names = _as_nonempty_string_tuple(self.feature_names, "feature_names")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique while preserving order")
        object.__setattr__(self, "feature_names", names)

        expected = self.calculate_sha256(
            schema_id=self.schema_id,
            schema_version=self.schema_version,
            feature_names=names,
        )
        if self.schema_sha256 is None:
            object.__setattr__(self, "schema_sha256", expected)
        else:
            _require_sha256(self.schema_sha256, "schema_sha256")
            if self.schema_sha256 != expected:
                raise ValueError(
                    "schema_sha256 does not authenticate the ordered feature schema"
                )

    @staticmethod
    def calculate_sha256(
        *,
        schema_id: str,
        schema_version: int,
        feature_names: Sequence[str],
    ) -> str:
        """Hash the canonical semantic content of an ordered schema."""

        canonical = json.dumps(
            {
                "feature_names": list(feature_names),
                "schema_id": schema_id,
                "schema_version": schema_version,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-safe representation."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "schema_sha256": self.schema_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_schema_sha256: str | None = None,
    ) -> OrderedFeatureSchema:
        """Restore a schema and verify its embedded and expected digests."""

        payload = _require_mapping(state, "ordered feature schema")
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "schema_id",
                "schema_version",
                "feature_names",
                "schema_sha256",
            },
            "ordered feature schema",
        )
        _require_contract_version(payload["contract_version"], cls.CONTRACT_VERSION)
        names = payload["feature_names"]
        if not isinstance(names, list):
            raise TypeError("ordered feature schema feature_names must be a JSON list")
        schema = cls(
            schema_id=payload["schema_id"],
            schema_version=payload["schema_version"],
            feature_names=tuple(names),
            schema_sha256=payload["schema_sha256"],
        )
        if (
            expected_schema_sha256 is not None
            and schema.schema_sha256 != expected_schema_sha256
        ):
            raise ValueError(
                "ordered feature schema does not match expected_schema_sha256"
            )
        return schema


@dataclass(frozen=True, slots=True)
class RegimeInference:
    """A point-in-time, causal regime inference shared across system layers.

    The filtered vector describes the state at ``observation_time`` using
    information available through that observation.  The one-step vector is
    the distribution for ``prediction_time`` conditioned on the same history.
    Smoothed probabilities are intentionally excluded because they use later
    observations and must never masquerade as an operational signal.
    """

    CONTRACT_VERSION: ClassVar[int] = 1

    entity_id: str
    sequence_id: str
    observation_time: datetime
    inference_time: datetime
    prediction_time: datetime
    model: ModelIdentity
    feature_schema: OrderedFeatureSchema
    state_ids: tuple[str, ...]
    filtered_probabilities: tuple[float, ...]
    one_step_probabilities: tuple[float, ...]
    dominant_state: str | None = None
    semantic_label: str | None = None
    log_predictive_density: float | None = None
    quality_flags: tuple[RegimeQualityFlag, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _require_identifier(self.entity_id, "entity_id")
        _require_identifier(self.sequence_id, "sequence_id")
        _require_aware_datetime(self.observation_time, "observation_time")
        _require_aware_datetime(self.inference_time, "inference_time")
        _require_aware_datetime(self.prediction_time, "prediction_time")
        if self.inference_time < self.observation_time:
            raise ValueError("inference_time cannot precede observation_time")
        if self.prediction_time <= self.inference_time:
            raise ValueError("prediction_time must be strictly after inference_time")

        if not isinstance(self.model, ModelIdentity):
            raise TypeError("model must be a ModelIdentity")
        if not isinstance(self.feature_schema, OrderedFeatureSchema):
            raise TypeError("feature_schema must be an OrderedFeatureSchema")
        if self.model.feature_schema_sha256 != self.feature_schema.schema_sha256:
            raise ValueError(
                "model feature_schema_sha256 does not match feature_schema"
            )

        states = _as_nonempty_string_tuple(self.state_ids, "state_ids")
        if len(set(states)) != len(states):
            raise ValueError("state_ids must be unique while preserving order")
        object.__setattr__(self, "state_ids", states)

        filtered = _as_simplex(
            self.filtered_probabilities,
            name="filtered_probabilities",
            expected_size=len(states),
        )
        one_step = _as_simplex(
            self.one_step_probabilities,
            name="one_step_probabilities",
            expected_size=len(states),
        )
        object.__setattr__(self, "filtered_probabilities", filtered)
        object.__setattr__(self, "one_step_probabilities", one_step)

        if self.dominant_state is not None:
            _require_identifier(self.dominant_state, "dominant_state")
            if self.dominant_state not in states:
                raise ValueError("dominant_state must be present in state_ids")
            dominant_probability = filtered[states.index(self.dominant_state)]
            if dominant_probability != max(filtered):
                raise ValueError(
                    "dominant_state must identify a maximum filtered probability"
                )
        if self.semantic_label is not None:
            _require_identifier(self.semantic_label, "semantic_label")

        if self.log_predictive_density is not None:
            log_density = _as_finite_float(
                self.log_predictive_density, "log_predictive_density"
            )
            object.__setattr__(self, "log_predictive_density", log_density)

        flags = _as_quality_flags(self.quality_flags)
        object.__setattr__(self, "quality_flags", flags)

    def probabilities_for(self, semantics: ProbabilitySemantics) -> tuple[float, ...]:
        """Return probabilities for a causal semantic, rejecting smoothing."""

        if semantics is ProbabilitySemantics.FILTERED:
            return self.filtered_probabilities
        if semantics is ProbabilitySemantics.ONE_STEP_PREDICTED:
            return self.one_step_probabilities
        if semantics is ProbabilitySemantics.SMOOTHED:
            raise ValueError("RegimeInference never contains smoothed probabilities")
        raise TypeError("semantics must be a ProbabilitySemantics")

    def state_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-safe representation."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            "entity_id": self.entity_id,
            "sequence_id": self.sequence_id,
            "observation_time": self.observation_time.isoformat(),
            "inference_time": self.inference_time.isoformat(),
            "prediction_time": self.prediction_time.isoformat(),
            "model": self.model.state_dict(),
            "feature_schema": self.feature_schema.state_dict(),
            "state_ids": list(self.state_ids),
            "probabilities": {
                ProbabilitySemantics.FILTERED.value: list(self.filtered_probabilities),
                ProbabilitySemantics.ONE_STEP_PREDICTED.value: list(
                    self.one_step_probabilities
                ),
            },
            "dominant_state": self.dominant_state,
            "semantic_label": self.semantic_label,
            "log_predictive_density": self.log_predictive_density,
            "quality_flags": [flag.value for flag in self.quality_flags],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_model_id: str | None = None,
        expected_artifact_sha256: str | None = None,
        expected_feature_schema_sha256: str | None = None,
    ) -> RegimeInference:
        """Restore an inference with strict nested schema and identity checks."""

        payload = _require_mapping(state, "regime inference")
        _require_exact_keys(
            payload,
            {
                "contract_version",
                "entity_id",
                "sequence_id",
                "observation_time",
                "inference_time",
                "prediction_time",
                "model",
                "feature_schema",
                "state_ids",
                "probabilities",
                "dominant_state",
                "semantic_label",
                "log_predictive_density",
                "quality_flags",
            },
            "regime inference",
        )
        _require_contract_version(payload["contract_version"], cls.CONTRACT_VERSION)

        model = ModelIdentity.from_state_dict(
            payload["model"],
            expected_model_id=expected_model_id,
            expected_artifact_sha256=expected_artifact_sha256,
        )
        feature_schema = OrderedFeatureSchema.from_state_dict(
            payload["feature_schema"],
            expected_schema_sha256=expected_feature_schema_sha256,
        )

        state_ids = payload["state_ids"]
        quality_flags = payload["quality_flags"]
        if not isinstance(state_ids, list):
            raise TypeError("regime inference state_ids must be a JSON list")
        if not isinstance(quality_flags, list):
            raise TypeError("regime inference quality_flags must be a JSON list")

        probabilities = _require_mapping(
            payload["probabilities"], "regime inference probabilities"
        )
        _require_exact_keys(
            probabilities,
            {
                ProbabilitySemantics.FILTERED.value,
                ProbabilitySemantics.ONE_STEP_PREDICTED.value,
            },
            "regime inference probabilities",
        )
        filtered = probabilities[ProbabilitySemantics.FILTERED.value]
        one_step = probabilities[ProbabilitySemantics.ONE_STEP_PREDICTED.value]
        if not isinstance(filtered, list) or not isinstance(one_step, list):
            raise TypeError("serialized probability vectors must be JSON lists")

        return cls(
            entity_id=payload["entity_id"],
            sequence_id=payload["sequence_id"],
            observation_time=_parse_datetime(
                payload["observation_time"], "observation_time"
            ),
            inference_time=_parse_datetime(payload["inference_time"], "inference_time"),
            prediction_time=_parse_datetime(
                payload["prediction_time"], "prediction_time"
            ),
            model=model,
            feature_schema=feature_schema,
            state_ids=tuple(state_ids),
            filtered_probabilities=tuple(filtered),
            one_step_probabilities=tuple(one_step),
            dominant_state=payload["dominant_state"],
            semantic_label=payload["semantic_label"],
            log_predictive_density=payload["log_predictive_density"],
            quality_flags=tuple(quality_flags),
        )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    state: Mapping[str, Any], expected: set[str], name: str
) -> None:
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{name} violates its frozen schema; missing={missing}, unknown={unknown}"
        )


def _require_contract_version(value: object, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"unsupported contract_version: {value!r}")


def _require_identifier(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty, whitespace-trimmed string")


def _require_sha256(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")


def _as_nonempty_string_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an ordered sequence of strings")
    items = tuple(value)
    if not items:
        raise ValueError(f"{name} cannot be empty")
    for index, item in enumerate(items):
        _require_identifier(item, f"{name}[{index}]")
    return items


def _as_finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite JSON number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_simplex(value: object, *, name: str, expected_size: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an ordered sequence of probabilities")
    probabilities = tuple(
        _as_finite_float(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(probabilities) != expected_size:
        raise ValueError(f"{name} length must match state_ids")
    if any(item < 0.0 or item > 1.0 for item in probabilities):
        raise ValueError(f"{name} must lie on the probability simplex")
    if abs(fsum(probabilities) - 1.0) > _SIMPLEX_TOLERANCE:
        raise ValueError(f"{name} must sum to one on the probability simplex")
    return probabilities


def _as_quality_flags(value: object) -> tuple[RegimeQualityFlag, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("quality_flags must be an ordered sequence")
    flags: list[RegimeQualityFlag] = []
    for item in value:
        if isinstance(item, RegimeQualityFlag):
            flag = item
        elif type(item) is str:
            try:
                flag = RegimeQualityFlag(item)
            except ValueError as exc:
                raise ValueError(f"unknown regime quality flag: {item!r}") from exc
        else:
            raise TypeError("quality_flags entries must be RegimeQualityFlag values")
        if flag in flags:
            raise ValueError("quality_flags cannot contain duplicates")
        flags.append(flag)
    return tuple(flags)


def _require_aware_datetime(value: object, name: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _parse_datetime(value: object, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"serialized {name} must be an ISO-8601 string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"serialized {name} is not valid ISO-8601") from exc
    _require_aware_datetime(result, name)
    if result.isoformat() != value:
        raise ValueError(f"serialized {name} must use canonical datetime.isoformat()")
    return result


__all__ = [
    "ModelIdentity",
    "OrderedFeatureSchema",
    "ProbabilitySemantics",
    "RegimeInference",
    "RegimeQualityFlag",
]
