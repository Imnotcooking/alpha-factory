"""Canonical causal, stateful filtering for fitted diagonal HMMs.

The session API is deliberately incapable of smoothing.  A reset begins at
the fitted initial distribution; a continuation begins only from an
externally verified :math:`P(S_{t+1} | F_t)` checkpoint for the same model,
entity, and declared sequence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import fsum, isfinite
from typing import Any

import numpy as np

from oqp.contracts.regime_state import (
    OrderedFeatureSchema,
    ProbabilitySemantics,
    RegimeInference,
    RegimeQualityFlag,
)

from .fitted import FittedDiagonalHMM
from .observations import ObservationSequence
from .serialization import sha256_json


FILTER_CHECKPOINT_VERSION = "causal_hmm_filter_checkpoint_v2"
_CHECKPOINT_FIELDS = frozenset(
    {
        "version",
        "model_id",
        "parameter_sha256",
        "feature_schema_sha256",
        "entity_id",
        "sequence_id",
        "origin_row_id",
        "observation_time",
        "probabilities",
        "semantics",
        "forecast_horizon_periods",
        "checkpoint_sha256",
    }
)


class CausalFilteringError(ValueError):
    """Raised when a causal recursion or continuation is invalid."""


class FilterStartMode(str, Enum):
    """Auditable origin of a causal filtering session."""

    RESET = "reset_to_fitted_initial_distribution"
    CONTINUE = "continue_from_verified_one_step_distribution"


@dataclass(frozen=True, slots=True)
class CausalFilterStep:
    """One observation update with unambiguous probability timing."""

    model: FittedDiagonalHMM
    entity_id: str
    sequence_id: str
    origin_row_id: str
    observation_time: datetime
    observation_prior_probabilities: tuple[float, ...]
    filtered_probabilities: tuple[float, ...]
    one_step_probabilities: tuple[float, ...]
    log_predictive_density: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, FittedDiagonalHMM):
            raise TypeError("model must be a FittedDiagonalHMM")
        _identifier(self.entity_id, "entity_id")
        _identifier(self.sequence_id, "sequence_id")
        _identifier(self.origin_row_id, "origin_row_id")
        _aware_datetime(self.observation_time, "observation_time")
        for name, probabilities in (
            ("observation_prior_probabilities", self.observation_prior_probabilities),
            ("filtered_probabilities", self.filtered_probabilities),
            ("one_step_probabilities", self.one_step_probabilities),
        ):
            parsed = _simplex_tuple(probabilities, name=name)
            if len(parsed) != self.model.n_states:
                raise ValueError(f"{name} state count does not match the model")
            object.__setattr__(self, name, parsed)
        if not isfinite(self.log_predictive_density):
            raise ValueError("log_predictive_density must be finite")

    def as_regime_inference(
        self,
        *,
        inference_time: datetime,
        prediction_time: datetime,
        semantic_label: str | None = None,
        quality_flags: Sequence[RegimeQualityFlag] = (),
    ) -> RegimeInference:
        """Adapt this numerical step to the cross-layer point-in-time contract."""

        dominant_index = int(np.argmax(self.filtered_probabilities))
        return RegimeInference(
            entity_id=self.entity_id,
            sequence_id=self.sequence_id,
            observation_time=self.observation_time,
            inference_time=inference_time,
            prediction_time=prediction_time,
            model=self.model.identity,
            feature_schema=self.model.feature_schema,
            state_ids=self.model.state_ids,
            filtered_probabilities=self.filtered_probabilities,
            one_step_probabilities=self.one_step_probabilities,
            dominant_state=self.model.state_ids[dominant_index],
            semantic_label=semantic_label,
            log_predictive_density=self.log_predictive_density,
            quality_flags=tuple(quality_flags),
        )


@dataclass(frozen=True, slots=True)
class FilterCheckpoint:
    r"""Content-addressed :math:`P(S_{t+1} | F_t)` continuation state.

    Authenticity requires comparing ``checkpoint_sha256`` with a digest held
    outside the checkpoint file, as enforced by the JSON loader.
    """

    version: str
    model_id: str
    parameter_sha256: str
    feature_schema_sha256: str
    entity_id: str
    sequence_id: str
    origin_row_id: str
    observation_time: datetime
    probabilities: tuple[float, ...]
    checkpoint_sha256: str
    semantics: ProbabilitySemantics = ProbabilitySemantics.ONE_STEP_PREDICTED
    forecast_horizon_periods: int = 1

    def __post_init__(self) -> None:
        if self.version != FILTER_CHECKPOINT_VERSION:
            raise ValueError("unsupported filter checkpoint version")
        _identifier(self.model_id, "model_id")
        _identifier(self.entity_id, "entity_id")
        _identifier(self.sequence_id, "sequence_id")
        _identifier(self.origin_row_id, "origin_row_id")
        _aware_datetime(self.observation_time, "observation_time")
        _sha256(self.parameter_sha256, "parameter_sha256")
        _sha256(self.feature_schema_sha256, "feature_schema_sha256")
        _sha256(self.checkpoint_sha256, "checkpoint_sha256")
        probabilities = _simplex_tuple(self.probabilities, name="probabilities")
        object.__setattr__(self, "probabilities", probabilities)
        if self.semantics is not ProbabilitySemantics.ONE_STEP_PREDICTED:
            raise ValueError(
                "a checkpoint must contain one-step predicted probabilities"
            )
        if self.forecast_horizon_periods != 1:
            raise ValueError("a checkpoint must have forecast horizon one")
        if sha256_json(self._payload()) != self.checkpoint_sha256:
            raise ValueError("checkpoint_sha256 does not authenticate the checkpoint")

    @classmethod
    def create(
        cls,
        *,
        model: FittedDiagonalHMM,
        entity_id: str,
        sequence_id: str,
        origin_row_id: str,
        observation_time: datetime,
        probabilities: Sequence[float],
    ) -> "FilterCheckpoint":
        if not isinstance(model, FittedDiagonalHMM):
            raise TypeError("model must be a FittedDiagonalHMM")
        _identifier(entity_id, "entity_id")
        _identifier(sequence_id, "sequence_id")
        _identifier(origin_row_id, "origin_row_id")
        _aware_datetime(observation_time, "observation_time")
        schema_hash = model.feature_schema.schema_sha256
        if schema_hash is None:  # guarded by OrderedFeatureSchema
            raise RuntimeError("feature schema has no authenticated digest")
        parsed = _simplex_tuple(probabilities, name="probabilities")
        if len(parsed) != model.n_states:
            raise ValueError("checkpoint state count does not match the model")
        payload = _checkpoint_payload(
            version=FILTER_CHECKPOINT_VERSION,
            model_id=model.model_id,
            parameter_sha256=model.parameter_sha256,
            feature_schema_sha256=schema_hash,
            entity_id=entity_id,
            sequence_id=sequence_id,
            origin_row_id=origin_row_id,
            observation_time=observation_time,
            probabilities=parsed,
            semantics=ProbabilitySemantics.ONE_STEP_PREDICTED,
            forecast_horizon_periods=1,
        )
        return cls(
            version=FILTER_CHECKPOINT_VERSION,
            model_id=model.model_id,
            parameter_sha256=model.parameter_sha256,
            feature_schema_sha256=schema_hash,
            entity_id=entity_id,
            sequence_id=sequence_id,
            origin_row_id=origin_row_id,
            observation_time=observation_time,
            probabilities=parsed,
            checkpoint_sha256=sha256_json(payload),
        )

    def _payload(self) -> dict[str, Any]:
        return _checkpoint_payload(
            version=self.version,
            model_id=self.model_id,
            parameter_sha256=self.parameter_sha256,
            feature_schema_sha256=self.feature_schema_sha256,
            entity_id=self.entity_id,
            sequence_id=self.sequence_id,
            origin_row_id=self.origin_row_id,
            observation_time=self.observation_time,
            probabilities=self.probabilities,
            semantics=self.semantics,
            forecast_horizon_periods=self.forecast_horizon_periods,
        )

    def state_dict(self) -> dict[str, Any]:
        return {**self._payload(), "checkpoint_sha256": self.checkpoint_sha256}

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_model_id: str,
        expected_parameter_sha256: str,
        expected_entity_id: str,
        expected_checkpoint_sha256: str,
    ) -> "FilterCheckpoint":
        if not isinstance(state, Mapping) or any(type(key) is not str for key in state):
            raise TypeError("filter checkpoint must be a string-keyed mapping")
        if set(state) != _CHECKPOINT_FIELDS:
            missing = sorted(_CHECKPOINT_FIELDS.difference(state))
            unknown = sorted(set(state).difference(_CHECKPOINT_FIELDS))
            raise ValueError(
                "filter checkpoint fields differ from the schema; "
                f"missing={missing}, unknown={unknown}"
            )
        probabilities = state["probabilities"]
        if not isinstance(probabilities, list):
            raise TypeError("checkpoint probabilities must be a JSON list")
        parsed_probabilities = tuple(
            _json_number(value, "probabilities[]") for value in probabilities
        )
        semantics_value = state["semantics"]
        if type(semantics_value) is not str:
            raise TypeError("checkpoint semantics must be a string")
        try:
            semantics = ProbabilitySemantics(semantics_value)
        except ValueError as exc:
            raise ValueError("unsupported checkpoint probability semantics") from exc
        if type(state["forecast_horizon_periods"]) is not int:
            raise TypeError("forecast_horizon_periods must be an integer")
        _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
        if state["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError(
                "checkpoint_sha256 differs from the independently trusted digest"
            )
        checkpoint = cls(
            version=_json_string(state["version"], "version"),
            model_id=_json_string(state["model_id"], "model_id"),
            parameter_sha256=_json_string(
                state["parameter_sha256"], "parameter_sha256"
            ),
            feature_schema_sha256=_json_string(
                state["feature_schema_sha256"], "feature_schema_sha256"
            ),
            entity_id=_json_string(state["entity_id"], "entity_id"),
            sequence_id=_json_string(state["sequence_id"], "sequence_id"),
            origin_row_id=_json_string(state["origin_row_id"], "origin_row_id"),
            observation_time=_parse_datetime(
                state["observation_time"], "observation_time"
            ),
            probabilities=parsed_probabilities,
            semantics=semantics,
            forecast_horizon_periods=state["forecast_horizon_periods"],
            checkpoint_sha256=_json_string(
                state["checkpoint_sha256"], "checkpoint_sha256"
            ),
        )
        if checkpoint.model_id != expected_model_id:
            raise ValueError("checkpoint model_id differs from expected_model_id")
        if checkpoint.entity_id != expected_entity_id:
            raise ValueError("checkpoint entity_id differs from expected_entity_id")
        _sha256(expected_parameter_sha256, "expected_parameter_sha256")
        if checkpoint.parameter_sha256 != expected_parameter_sha256:
            raise ValueError(
                "checkpoint parameters differ from expected_parameter_sha256"
            )
        return checkpoint

    def require_compatible(
        self,
        model: FittedDiagonalHMM,
        *,
        entity_id: str,
        sequence_id: str,
    ) -> None:
        """Fail closed before carrying state into another model or sequence."""

        if self.model_id != model.model_id:
            raise CausalFilteringError("checkpoint model_id does not match")
        if self.parameter_sha256 != model.parameter_sha256:
            raise CausalFilteringError("checkpoint parameter digest does not match")
        if self.feature_schema_sha256 != model.feature_schema.schema_sha256:
            raise CausalFilteringError("checkpoint feature schema does not match")
        if self.entity_id != entity_id:
            raise CausalFilteringError("checkpoint cannot cross an entity boundary")
        if self.sequence_id != sequence_id:
            raise CausalFilteringError(
                "checkpoint cannot cross a declared sequence boundary"
            )
        if len(self.probabilities) != model.n_states:
            raise CausalFilteringError("checkpoint state count does not match")


class CausalFilterSession:
    """Mutable online filter state bound to one model and one sequence."""

    __slots__ = (
        "_model",
        "_entity_id",
        "_sequence_id",
        "_prior",
        "_last_origin_row_id",
        "_last_observation_time",
        "_start_mode",
        "_updates",
    )

    def __init__(
        self,
        *,
        model: FittedDiagonalHMM,
        entity_id: str,
        sequence_id: str,
        checkpoint: FilterCheckpoint | None = None,
    ) -> None:
        if not isinstance(model, FittedDiagonalHMM):
            raise TypeError("model must be a FittedDiagonalHMM")
        _identifier(entity_id, "entity_id")
        _identifier(sequence_id, "sequence_id")
        if checkpoint is None:
            prior = model.initial_probabilities
            start_mode = FilterStartMode.RESET
            last_origin_row_id = None
            last_observation_time = None
        else:
            if not isinstance(checkpoint, FilterCheckpoint):
                raise TypeError("checkpoint must be a FilterCheckpoint")
            checkpoint.require_compatible(
                model,
                entity_id=entity_id,
                sequence_id=sequence_id,
            )
            prior = checkpoint.probabilities
            start_mode = FilterStartMode.CONTINUE
            last_origin_row_id = checkpoint.origin_row_id
            last_observation_time = checkpoint.observation_time
        parsed = _simplex_tuple(prior, name="prior")
        if len(parsed) != model.n_states:
            raise ValueError("prior state count does not match the model")
        self._model = model
        self._entity_id = entity_id
        self._sequence_id = sequence_id
        self._prior = np.asarray(parsed, dtype=np.float64)
        self._last_origin_row_id = last_origin_row_id
        self._last_observation_time = last_observation_time
        self._start_mode = start_mode
        self._updates = 0

    @classmethod
    def reset(
        cls,
        model: FittedDiagonalHMM,
        *,
        entity_id: str,
        sequence_id: str,
    ) -> "CausalFilterSession":
        """Start a new declared sequence from the fitted initial distribution."""

        return cls(
            model=model,
            entity_id=entity_id,
            sequence_id=sequence_id,
        )

    @classmethod
    def continue_from_checkpoint(
        cls,
        model: FittedDiagonalHMM,
        checkpoint: FilterCheckpoint,
        *,
        entity_id: str,
        sequence_id: str,
    ) -> "CausalFilterSession":
        """Continue the same entity sequence from a verified one-step state."""

        return cls(
            model=model,
            entity_id=entity_id,
            sequence_id=sequence_id,
            checkpoint=checkpoint,
        )

    @property
    def model(self) -> FittedDiagonalHMM:
        return self._model

    @property
    def sequence_id(self) -> str:
        return self._sequence_id

    @property
    def entity_id(self) -> str:
        return self._entity_id

    @property
    def start_mode(self) -> FilterStartMode:
        return self._start_mode

    def update(
        self,
        observation: Sequence[float] | np.ndarray,
        *,
        feature_schema: OrderedFeatureSchema,
        origin_row_id: str,
        observation_time: datetime,
    ) -> CausalFilterStep:
        """Consume exactly one row and advance the causal one-step prior."""

        _identifier(origin_row_id, "origin_row_id")
        if not isinstance(feature_schema, OrderedFeatureSchema):
            raise TypeError("feature_schema must be an OrderedFeatureSchema")
        if feature_schema.schema_sha256 != self._model.feature_schema.schema_sha256:
            raise CausalFilteringError("observation feature schema does not match")
        _aware_datetime(observation_time, "observation_time")
        if origin_row_id == self._last_origin_row_id:
            raise CausalFilteringError("the latest origin_row_id cannot be replayed")
        if (
            self._last_observation_time is not None
            and observation_time <= self._last_observation_time
        ):
            raise CausalFilteringError(
                "observation_time must advance strictly within a session"
            )
        try:
            row = np.asarray(observation, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CausalFilteringError("observation must be numeric") from exc
        if row.ndim != 1 or row.shape[0] != self._model.n_features:
            raise CausalFilteringError(
                "observation must be a vector matching the model feature width"
            )
        if not np.isfinite(row).all():
            raise CausalFilteringError("observation must contain finite values")

        emissions = self._model.log_emission_probabilities(
            row[None, :],
            feature_schema=feature_schema,
        )[0]
        with np.errstate(divide="ignore"):
            log_prior = np.log(self._prior)
            log_transition = np.log(self._model.transition_matrix)
        log_joint = log_prior + emissions
        log_density = _logsumexp(log_joint)
        if not isfinite(log_density):
            raise CausalFilteringError(
                "observation has no finite-probability state path"
            )
        log_filtered = log_joint - log_density
        filtered = _probabilities_from_normalized_logs(log_filtered)
        log_next = np.asarray(
            [
                _logsumexp(log_filtered + log_transition[:, destination])
                for destination in range(self._model.n_states)
            ],
            dtype=np.float64,
        )
        next_probabilities = _probabilities_from_normalized_logs(
            log_next - _logsumexp(log_next)
        )
        step = CausalFilterStep(
            model=self._model,
            entity_id=self._entity_id,
            sequence_id=self._sequence_id,
            origin_row_id=origin_row_id,
            observation_time=observation_time,
            observation_prior_probabilities=_as_tuple(self._prior),
            filtered_probabilities=_as_tuple(filtered),
            one_step_probabilities=_as_tuple(next_probabilities),
            log_predictive_density=float(log_density),
        )
        self._prior = next_probabilities
        self._last_origin_row_id = origin_row_id
        self._last_observation_time = observation_time
        self._updates += 1
        return step

    def infer(
        self,
        observation: Sequence[float] | np.ndarray,
        *,
        feature_schema: OrderedFeatureSchema,
        origin_row_id: str,
        observation_time: datetime,
        inference_time: datetime,
        prediction_time: datetime,
        semantic_label: str | None = None,
        quality_flags: Sequence[RegimeQualityFlag] = (),
    ) -> RegimeInference:
        """Consume one row and emit the dependency-light cross-layer contract."""

        first_reset = self._updates == 0 and self._start_mode is FilterStartMode.RESET
        snapshot = (
            self._prior.copy(),
            self._last_origin_row_id,
            self._last_observation_time,
            self._updates,
        )
        step = self.update(
            observation,
            feature_schema=feature_schema,
            origin_row_id=origin_row_id,
            observation_time=observation_time,
        )
        try:
            flags = tuple(quality_flags)
            if first_reset and RegimeQualityFlag.STATE_RESET not in flags:
                flags = (*flags, RegimeQualityFlag.STATE_RESET)
            return step.as_regime_inference(
                inference_time=inference_time,
                prediction_time=prediction_time,
                semantic_label=semantic_label,
                quality_flags=flags,
            )
        except BaseException:
            (
                self._prior,
                self._last_origin_row_id,
                self._last_observation_time,
                self._updates,
            ) = snapshot
            raise

    def checkpoint(self) -> FilterCheckpoint:
        """Freeze the current one-step prior for an exact later continuation."""

        if self._last_origin_row_id is None or self._last_observation_time is None:
            raise CausalFilteringError("cannot checkpoint before the first update")
        return FilterCheckpoint.create(
            model=self._model,
            entity_id=self._entity_id,
            sequence_id=self._sequence_id,
            origin_row_id=self._last_origin_row_id,
            observation_time=self._last_observation_time,
            probabilities=self._prior,
        )


@dataclass(frozen=True, slots=True)
class SequenceFilterResult:
    """Completed forward-only recursion for one explicit sequence."""

    sequence: ObservationSequence
    steps: tuple[CausalFilterStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, ObservationSequence):
            raise TypeError("sequence must be an ObservationSequence")
        if len(self.steps) != self.sequence.n_observations:
            raise ValueError("one causal step is required per observation")
        if tuple(step.origin_row_id for step in self.steps) != self.sequence.row_ids:
            raise ValueError("filter step row IDs must preserve observation order")
        if tuple(step.observation_time for step in self.steps) != (
            self.sequence.observation_times
        ):
            raise ValueError("filter step timestamps must preserve observation order")
        if any(step.entity_id != self.sequence.entity_id for step in self.steps):
            raise ValueError("filter steps must remain within the sequence entity")

    @property
    def log_likelihood(self) -> float:
        return fsum(step.log_predictive_density for step in self.steps)


def filter_observation_sequence(
    model: FittedDiagonalHMM,
    sequence: ObservationSequence,
    *,
    checkpoint: FilterCheckpoint | None = None,
) -> SequenceFilterResult:
    """Filter a reset sequence or a same-sequence continuation chunk."""

    if sequence.feature_schema.schema_sha256 != model.feature_schema.schema_sha256:
        raise ValueError("observation feature schema does not match the model")
    session = (
        CausalFilterSession.reset(
            model,
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id,
        )
        if checkpoint is None
        else CausalFilterSession.continue_from_checkpoint(
            model,
            checkpoint,
            entity_id=sequence.entity_id,
            sequence_id=sequence.sequence_id,
        )
    )
    steps = tuple(
        session.update(
            row,
            feature_schema=sequence.feature_schema,
            origin_row_id=row_id,
            observation_time=observation_time,
        )
        for row_id, observation_time, row in zip(
            sequence.row_ids,
            sequence.observation_times,
            sequence.values,
            strict=True,
        )
    )
    return SequenceFilterResult(sequence=sequence, steps=steps)


def _checkpoint_payload(
    *,
    version: str,
    model_id: str,
    parameter_sha256: str,
    feature_schema_sha256: str,
    entity_id: str,
    sequence_id: str,
    origin_row_id: str,
    observation_time: datetime,
    probabilities: Sequence[float],
    semantics: ProbabilitySemantics,
    forecast_horizon_periods: int,
) -> dict[str, Any]:
    return {
        "version": version,
        "model_id": model_id,
        "parameter_sha256": parameter_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "entity_id": entity_id,
        "sequence_id": sequence_id,
        "origin_row_id": origin_row_id,
        "observation_time": observation_time.isoformat(),
        "probabilities": list(probabilities),
        "semantics": semantics.value,
        "forecast_horizon_periods": forecast_horizon_periods,
    }


def _simplex_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        parsed = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if len(parsed) < 2:
        raise ValueError(f"{name} requires at least two states")
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in parsed):
        raise ValueError(f"{name} values must be finite and lie in [0, 1]")
    if abs(fsum(parsed) - 1.0) > 1e-10:
        raise ValueError(f"{name} must sum to one")
    total = fsum(parsed)
    return tuple(value / total for value in parsed)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not isfinite(maximum):
        return maximum
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def _probabilities_from_normalized_logs(values: np.ndarray) -> np.ndarray:
    probabilities = np.exp(values)
    total = float(probabilities.sum())
    if not isfinite(total) or total <= 0.0:
        raise CausalFilteringError("log probabilities could not be normalized")
    probabilities /= total
    if not np.isfinite(probabilities).all():
        raise CausalFilteringError("filter recursion produced non-finite probabilities")
    return probabilities


def _as_tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _aware_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _parse_datetime(value: Any, name: str) -> datetime:
    if type(value) is not str:
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be valid ISO-8601") from exc
    _aware_datetime(parsed, name)
    if parsed.isoformat() != value:
        raise ValueError(f"{name} must use canonical datetime.isoformat() encoding")
    return parsed


def _json_string(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty JSON string")
    return value


def _json_number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a JSON number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "CausalFilterSession",
    "CausalFilterStep",
    "CausalFilteringError",
    "FILTER_CHECKPOINT_VERSION",
    "FilterCheckpoint",
    "FilterStartMode",
    "SequenceFilterResult",
    "filter_observation_sequence",
]
