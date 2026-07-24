"""Canonical sequence-aware observations for shared regime models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from oqp.contracts.regime_state import OrderedFeatureSchema


def freeze_float_array(
    values: Any,
    *,
    ndim: int,
    name: str,
    require_finite: bool = True,
) -> np.ndarray:
    """Return a C-contiguous float64 array backed by immutable bytes.

    A normal read-only NumPy array that owns its memory can have its writable
    flag re-enabled.  Rebuilding from an immutable ``bytes`` buffer prevents
    that escape hatch and makes frozen model/observation dataclasses genuinely
    immutable at their numerical boundary.
    """

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a rectangular numeric array") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must have exactly {ndim} dimensions")
    if any(size < 1 for size in array.shape):
        raise ValueError(f"{name} dimensions must be non-empty")
    if require_finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return frozen.reshape(contiguous.shape)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_tuple(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class ObservationSequence:
    """One contiguous entity-local sequence with an explicit reset boundary."""

    sequence_id: str
    entity_id: str
    row_ids: tuple[str, ...]
    observation_times: tuple[datetime, ...]
    feature_schema: OrderedFeatureSchema
    values: np.ndarray

    def __post_init__(self) -> None:
        _identifier(self.sequence_id, "sequence_id")
        _identifier(self.entity_id, "entity_id")
        if not isinstance(self.feature_schema, OrderedFeatureSchema):
            raise TypeError("feature_schema must be an OrderedFeatureSchema")
        row_ids = _string_tuple(self.row_ids, "row_ids")
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("row_ids must be unique within a sequence")
        observation_times = tuple(self.observation_times)
        if len(observation_times) != len(row_ids):
            raise ValueError("observation_times must align one-for-one with row_ids")
        for index, timestamp in enumerate(observation_times):
            _aware_datetime(timestamp, f"observation_times[{index}]")
        if any(
            current <= previous
            for previous, current in zip(
                observation_times,
                observation_times[1:],
                strict=False,
            )
        ):
            raise ValueError("observation_times must be strictly increasing")
        matrix = freeze_float_array(self.values, ndim=2, name="values")
        if matrix.shape[0] != len(row_ids):
            raise ValueError("values must contain one row for every row_id")
        if matrix.shape[1] != len(self.feature_schema.feature_names):
            raise ValueError("values width must match the ordered feature schema")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "observation_times", observation_times)
        object.__setattr__(self, "values", matrix)

    @property
    def n_observations(self) -> int:
        return self.values.shape[0]

    @property
    def n_features(self) -> int:
        return self.values.shape[1]


def _aware_datetime(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class ObservationBatch:
    """Independent sequences that share exactly one ordered feature schema."""

    sequences: tuple[ObservationSequence, ...]

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        if not sequences:
            raise ValueError("an observation batch requires at least one sequence")
        if any(not isinstance(item, ObservationSequence) for item in sequences):
            raise TypeError("sequences must contain ObservationSequence objects")
        identifiers = [item.sequence_id for item in sequences]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sequence_id values must be unique within a batch")
        expected_hash = sequences[0].feature_schema.schema_sha256
        if any(
            item.feature_schema.schema_sha256 != expected_hash for item in sequences
        ):
            raise ValueError("all sequences must use the same ordered feature schema")
        object.__setattr__(self, "sequences", sequences)

    @property
    def feature_schema(self) -> OrderedFeatureSchema:
        return self.sequences[0].feature_schema

    @property
    def n_features(self) -> int:
        return self.sequences[0].n_features

    @property
    def n_observations(self) -> int:
        return sum(sequence.n_observations for sequence in self.sequences)


__all__ = ["ObservationBatch", "ObservationSequence", "freeze_float_array"]
