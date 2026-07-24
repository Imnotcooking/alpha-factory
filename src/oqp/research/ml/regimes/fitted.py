"""Canonical fitted parameters and emission densities for diagonal HMMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, lgamma
from typing import Any

import numpy as np

from oqp.contracts.regime_state import ModelIdentity, OrderedFeatureSchema

from .base import HMMFamily
from .observations import ObservationSequence, freeze_float_array
from .serialization import sha256_json


FITTED_DIAGONAL_HMM_VERSION = "shared_diagonal_hmm_v1"
_STATE_FIELDS = frozenset(
    {
        "contract_version",
        "model_id",
        "model_version",
        "family",
        "training_run_id",
        "preprocessing_artifact_sha256",
        "feature_schema",
        "state_ids",
        "initial_probabilities",
        "transition_matrix",
        "mixture_weights",
        "means",
        "diagonal_scales",
        "student_t_degrees_of_freedom",
        "parameter_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class FittedDiagonalHMM:
    """Authenticated parameters usable by research and online inference.

    Emission arrays have canonical shape ``K x M x D``.  Gaussian scales are
    variances.  Student-t scales are the diagonal scale-matrix entries, whose
    covariance is ``nu / (nu - 2)`` times the scale.  The NumPy arrays are
    backed by immutable byte buffers, so callers cannot mutate a fitted model
    after its parameter digest has been calculated.  When supplied, the
    preprocessing artifact digest is part of that parameter digest: the same
    numerical HMM cannot be silently paired with another preprocessing fit.
    """

    model_id: str
    family: HMMFamily
    feature_schema: OrderedFeatureSchema
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    diagonal_scales: np.ndarray
    mixture_weights: np.ndarray | None = None
    state_ids: tuple[str, ...] = ()
    student_t_degrees_of_freedom: float | None = None
    training_run_id: str | None = None
    preprocessing_artifact_sha256: str | None = None
    model_version: str = FITTED_DIAGONAL_HMM_VERSION
    _parameter_sha256: str = field(init=False, repr=False)

    CONTRACT_VERSION = 1

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        _identifier(self.model_version, "model_version")
        if self.training_run_id is not None:
            _identifier(self.training_run_id, "training_run_id")
        if self.preprocessing_artifact_sha256 is not None:
            _sha256_string(
                self.preprocessing_artifact_sha256,
                "preprocessing_artifact_sha256",
            )
        if not isinstance(self.family, HMMFamily):
            raise TypeError("family must be an HMMFamily")
        if not isinstance(self.feature_schema, OrderedFeatureSchema):
            raise TypeError("feature_schema must be an OrderedFeatureSchema")

        initial = _probability_vector(
            self.initial_probabilities,
            name="initial_probabilities",
            minimum_size=2,
        )
        state_count = initial.shape[0]
        transition = freeze_float_array(
            self.transition_matrix, ndim=2, name="transition_matrix"
        )
        if transition.shape != (state_count, state_count):
            raise ValueError("transition_matrix must be K x K")
        transition = _probability_rows(transition, name="transition_matrix")

        means_input = np.asarray(self.means, dtype=np.float64)
        scales_input = np.asarray(self.diagonal_scales, dtype=np.float64)
        if means_input.ndim == 2:
            means_input = means_input[:, None, :]
        if scales_input.ndim == 2:
            scales_input = scales_input[:, None, :]
        means = freeze_float_array(means_input, ndim=3, name="means")
        scales = freeze_float_array(scales_input, ndim=3, name="diagonal_scales")
        if means.shape != scales.shape:
            raise ValueError("means and diagonal_scales must have identical geometry")
        if means.shape[0] != state_count:
            raise ValueError("emission state count must match initial probabilities")
        if means.shape[2] != len(self.feature_schema.feature_names):
            raise ValueError("emission width must match the ordered feature schema")
        if np.any(scales <= 0.0):
            raise ValueError("diagonal_scales must be strictly positive")

        mixture_count = means.shape[1]
        if self.mixture_weights is None:
            if mixture_count != 1:
                raise ValueError("multi-component emissions require mixture_weights")
            weights_input = np.ones((state_count, 1), dtype=np.float64)
        else:
            weights_input = self.mixture_weights
        weights = freeze_float_array(weights_input, ndim=2, name="mixture_weights")
        if weights.shape != (state_count, mixture_count):
            raise ValueError("mixture_weights must have shape K x M")
        weights = _probability_rows(weights, name="mixture_weights")

        if self.family is HMMFamily.GAUSSIAN_MIXTURE:
            if mixture_count < 2:
                raise ValueError("a GMM-HMM requires at least two components")
        elif mixture_count != 1:
            raise ValueError("non-mixture HMM families require one component")

        degrees = self.student_t_degrees_of_freedom
        if self.family is HMMFamily.STUDENT_T:
            if degrees is None or not isfinite(degrees) or degrees <= 2.0:
                raise ValueError(
                    "Student-t emissions require finite degrees of freedom above two"
                )
            object.__setattr__(self, "student_t_degrees_of_freedom", float(degrees))
        elif degrees is not None:
            raise ValueError("only Student-t HMMs expose degrees of freedom")

        if self.state_ids:
            states = tuple(self.state_ids)
            if len(states) != state_count:
                raise ValueError("state_ids must contain one ID per state")
            if any(type(item) is not str or not item for item in states):
                raise ValueError("state_ids must contain non-empty strings")
            if len(set(states)) != len(states):
                raise ValueError("state_ids must be unique")
        else:
            states = tuple(f"state_{index}" for index in range(state_count))

        object.__setattr__(self, "initial_probabilities", initial)
        object.__setattr__(self, "transition_matrix", transition)
        object.__setattr__(self, "mixture_weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "diagonal_scales", scales)
        object.__setattr__(self, "state_ids", states)
        object.__setattr__(
            self, "_parameter_sha256", sha256_json(self._parameter_payload())
        )

    @property
    def n_states(self) -> int:
        return self.initial_probabilities.shape[0]

    @property
    def n_mixtures(self) -> int:
        return self.means.shape[1]

    @property
    def n_features(self) -> int:
        return self.means.shape[2]

    @property
    def parameter_sha256(self) -> str:
        return self._parameter_sha256

    @property
    def identity(self) -> ModelIdentity:
        """Return the dependency-light identity consumed outside research."""

        schema_hash = self.feature_schema.schema_sha256
        if schema_hash is None:  # guarded by OrderedFeatureSchema
            raise RuntimeError("feature schema has no authenticated digest")
        return ModelIdentity(
            model_id=self.model_id,
            model_family=self.family.value,
            model_version=self.model_version,
            artifact_sha256=self.parameter_sha256,
            feature_schema_sha256=schema_hash,
            training_run_id=self.training_run_id,
        )

    def log_emission_probabilities(
        self,
        observations: ObservationSequence | Sequence[Sequence[float]] | np.ndarray,
        *,
        feature_schema: OrderedFeatureSchema | None = None,
    ) -> np.ndarray:
        """Return an immutable ``T x K`` matrix of state log densities."""

        return log_emission_probabilities(
            self,
            observations,
            feature_schema=feature_schema,
        )

    def _parameter_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "family": self.family.value,
            "training_run_id": self.training_run_id,
            "preprocessing_artifact_sha256": (self.preprocessing_artifact_sha256),
            "feature_schema": self.feature_schema.state_dict(),
            "state_ids": list(self.state_ids),
            "initial_probabilities": self.initial_probabilities.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "mixture_weights": self.mixture_weights.tolist(),
            "means": self.means.tolist(),
            "diagonal_scales": self.diagonal_scales.tolist(),
            "student_t_degrees_of_freedom": self.student_t_degrees_of_freedom,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe state including its parameter digest."""

        return {
            "contract_version": self.CONTRACT_VERSION,
            **self._parameter_payload(),
            "parameter_sha256": self.parameter_sha256,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        expected_model_id: str | None = None,
        expected_parameter_sha256: str | None = None,
    ) -> "FittedDiagonalHMM":
        """Strictly reconstruct and authenticate a fitted diagonal HMM."""

        if not isinstance(state, Mapping) or any(type(key) is not str for key in state):
            raise TypeError("fitted HMM state must be a string-keyed mapping")
        if set(state) != _STATE_FIELDS:
            missing = sorted(_STATE_FIELDS.difference(state))
            unknown = sorted(set(state).difference(_STATE_FIELDS))
            raise ValueError(
                "fitted HMM state fields differ from the schema; "
                f"missing={missing}, unknown={unknown}"
            )
        if type(state["contract_version"]) is not int or state["contract_version"] != 1:
            raise ValueError("unsupported fitted HMM contract_version")
        family_value = state["family"]
        if type(family_value) is not str:
            raise TypeError("family must be a string")
        try:
            family = HMMFamily(family_value)
        except ValueError as exc:
            raise ValueError(f"unsupported HMM family: {family_value}") from exc
        schema = OrderedFeatureSchema.from_state_dict(state["feature_schema"])
        state_ids = state["state_ids"]
        if not isinstance(state_ids, list):
            raise TypeError("state_ids must be a JSON list")
        degrees = state["student_t_degrees_of_freedom"]
        if degrees is not None:
            degrees = _json_number(degrees, "student_t_degrees_of_freedom")
        model = cls(
            model_id=_json_string(state["model_id"], "model_id"),
            model_version=_json_string(state["model_version"], "model_version"),
            family=family,
            training_run_id=(
                None
                if state["training_run_id"] is None
                else _json_string(state["training_run_id"], "training_run_id")
            ),
            preprocessing_artifact_sha256=(
                None
                if state["preprocessing_artifact_sha256"] is None
                else _sha256_string(
                    state["preprocessing_artifact_sha256"],
                    "preprocessing_artifact_sha256",
                )
            ),
            feature_schema=schema,
            state_ids=tuple(state_ids),
            initial_probabilities=_numeric_json_array(
                state["initial_probabilities"], depth=1, name="initial_probabilities"
            ),
            transition_matrix=_numeric_json_array(
                state["transition_matrix"], depth=2, name="transition_matrix"
            ),
            mixture_weights=_numeric_json_array(
                state["mixture_weights"], depth=2, name="mixture_weights"
            ),
            means=_numeric_json_array(state["means"], depth=3, name="means"),
            diagonal_scales=_numeric_json_array(
                state["diagonal_scales"], depth=3, name="diagonal_scales"
            ),
            student_t_degrees_of_freedom=degrees,
        )
        embedded_hash = _sha256_string(state["parameter_sha256"], "parameter_sha256")
        if model.parameter_sha256 != embedded_hash:
            raise ValueError("parameter_sha256 does not authenticate fitted parameters")
        if expected_model_id is not None and model.model_id != expected_model_id:
            raise ValueError("fitted HMM model_id differs from expected_model_id")
        if (
            expected_parameter_sha256 is not None
            and model.parameter_sha256 != expected_parameter_sha256
        ):
            raise ValueError(
                "fitted HMM parameters differ from expected_parameter_sha256"
            )
        return model


def log_emission_probabilities(
    model: FittedDiagonalHMM,
    observations: ObservationSequence | Sequence[Sequence[float]] | np.ndarray,
    *,
    feature_schema: OrderedFeatureSchema | None = None,
) -> np.ndarray:
    """Evaluate Gaussian, GMM, or Student-t diagonal state densities."""

    if not isinstance(model, FittedDiagonalHMM):
        raise TypeError("model must be a FittedDiagonalHMM")
    if isinstance(observations, ObservationSequence):
        if feature_schema is not None:
            raise ValueError(
                "feature_schema must come from ObservationSequence, not a second source"
            )
        if (
            observations.feature_schema.schema_sha256
            != model.feature_schema.schema_sha256
        ):
            raise ValueError("observation feature schema does not match the model")
        values = observations.values
    else:
        if not isinstance(feature_schema, OrderedFeatureSchema):
            raise TypeError(
                "raw observations require an authenticated OrderedFeatureSchema"
            )
        if feature_schema.schema_sha256 != model.feature_schema.schema_sha256:
            raise ValueError("observation feature schema does not match the model")
        values = freeze_float_array(observations, ndim=2, name="observations")
    if values.shape[1] != model.n_features:
        raise ValueError("observation width does not match the fitted model")

    differences = values[:, None, None, :] - model.means[None, :, :, :]
    dimension = model.n_features
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        if model.family is HMMFamily.STUDENT_T:
            degrees = model.student_t_degrees_of_freedom
            if degrees is None:  # guarded at construction
                raise RuntimeError("Student-t model has no degrees of freedom")
            squared_distance = np.sum(
                differences**2 / model.diagonal_scales[None, :, :, :], axis=3
            )
            constant = (
                lgamma((degrees + dimension) / 2.0)
                - lgamma(degrees / 2.0)
                - 0.5 * dimension * np.log(degrees * np.pi)
            )
            component_logs = (
                constant
                - 0.5 * np.log(model.diagonal_scales).sum(axis=2)[None, :, :]
                - 0.5 * (degrees + dimension) * np.log1p(squared_distance / degrees)
            )
        else:
            component_logs = -0.5 * (
                dimension * np.log(2.0 * np.pi)
                + np.log(model.diagonal_scales).sum(axis=2)[None, :, :]
                + np.sum(differences**2 / model.diagonal_scales[None, :, :, :], axis=3)
            )
        component_logs = component_logs + np.log(model.mixture_weights)[None, :, :]
        state_logs = np.logaddexp.reduce(component_logs, axis=2)
    if not np.isfinite(state_logs).all():
        raise FloatingPointError("emission calculation produced non-finite values")
    return freeze_float_array(state_logs, ndim=2, name="log_emission_probabilities")


def _probability_vector(values: Any, *, name: str, minimum_size: int = 1) -> np.ndarray:
    array = freeze_float_array(values, ndim=1, name=name)
    if array.shape[0] < minimum_size:
        raise ValueError(f"{name} has too few entries")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1]")
    total = float(array.sum())
    if not np.isclose(total, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must sum to one")
    return freeze_float_array(array / total, ndim=1, name=name)


def _probability_rows(values: Any, *, name: str) -> np.ndarray:
    array = freeze_float_array(values, ndim=2, name=name)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1]")
    totals = array.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"every {name} row must sum to one")
    return freeze_float_array(array / totals[:, None], ndim=2, name=name)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


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


def _numeric_json_array(value: Any, *, depth: int, name: str) -> Any:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    if depth == 1:
        return [_json_number(item, f"{name}[]") for item in value]
    return [
        _numeric_json_array(item, depth=depth - 1, name=f"{name}[]") for item in value
    ]


def _sha256_string(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "FITTED_DIAGONAL_HMM_VERSION",
    "FittedDiagonalHMM",
    "log_emission_probabilities",
]
