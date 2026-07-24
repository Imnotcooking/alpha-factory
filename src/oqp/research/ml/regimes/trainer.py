"""Deterministic expectation-maximisation for shared diagonal HMMs.

The estimator accepts an :class:`~oqp.research.ml.regimes.observations.ObservationBatch`
whose members are independent sequences.  Forward-backward is restarted from
the initial distribution for every member, so no transition is manufactured
across an entity, contract, or data-gap boundary.

The numerical choices intentionally mirror the frozen Paper 01 estimator while
remaining independent of that package: seeded k-means++ initialization,
log-space forward-backward, fixed-degree Student-t updates, deterministic
restarts, and highest-likelihood restart selection with a lowest-index tie
break.  The implementation uses only NumPy and private random generators.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, lgamma
from typing import Any

import numpy as np

from .base import DiagonalHMMConfig, HMMFamily
from .fitted import FittedDiagonalHMM
from .observations import ObservationBatch


SHARED_HMM_TRAINER_VERSION = "shared_diagonal_hmm_em_v1"
_MAX_SEED = (2**32) - 1


class HMMTrainingFailure(str, Enum):
    """Machine-readable reason that a restart was not eligible."""

    NON_CONVERGENCE = "non_convergence"
    NUMERICAL_ERROR = "numerical_error"
    OCCUPANCY_BELOW_FLOOR = "occupancy_below_floor"


@dataclass(frozen=True, slots=True)
class DiagonalHMMTrainingControls:
    """Backend-specific EM, restart, and acceptance controls.

    ``DiagonalHMMConfig`` declares the fitted-model geometry.  This separate
    object declares how that model is estimated, allowing one geometry to be
    evaluated under an explicitly versioned training policy.
    """

    n_restarts: int = 20
    max_iterations: int = 500
    tolerance_per_observation: float = 1e-6
    minimum_state_occupancy: float = 0.05
    random_seed: int = 42
    initial_self_transition_probability: float = 0.90
    probability_pseudocount: float = 1e-3
    maximum_per_observation_decrease: float = 1e-10

    def __post_init__(self) -> None:
        for name in ("n_restarts", "max_iterations", "random_seed"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.n_restarts < 1 or self.max_iterations < 1:
            raise ValueError("n_restarts and max_iterations must be positive")
        if not 0 <= self.random_seed <= _MAX_SEED:
            raise ValueError(f"random_seed must lie in [0, {_MAX_SEED}]")
        if (
            not isfinite(self.tolerance_per_observation)
            or self.tolerance_per_observation <= 0.0
        ):
            raise ValueError("tolerance_per_observation must be finite and positive")
        if (
            not isfinite(self.minimum_state_occupancy)
            or not 0.0 <= self.minimum_state_occupancy < 1.0
        ):
            raise ValueError("minimum_state_occupancy must lie in [0, 1)")
        if (
            not isfinite(self.initial_self_transition_probability)
            or not 0.0 < self.initial_self_transition_probability < 1.0
        ):
            raise ValueError(
                "initial_self_transition_probability must lie strictly in (0, 1)"
            )
        if (
            not isfinite(self.probability_pseudocount)
            or self.probability_pseudocount <= 0.0
        ):
            raise ValueError("probability_pseudocount must be finite and positive")
        if (
            not isfinite(self.maximum_per_observation_decrease)
            or self.maximum_per_observation_decrease < 0.0
        ):
            raise ValueError(
                "maximum_per_observation_decrease must be finite and non-negative"
            )

    def state_dict(self) -> dict[str, Any]:
        """Return every numerical training choice in canonical form."""

        return {
            "implementation_version": SHARED_HMM_TRAINER_VERSION,
            "n_restarts": self.n_restarts,
            "max_iterations": self.max_iterations,
            "tolerance_per_observation": self.tolerance_per_observation,
            "minimum_state_occupancy": self.minimum_state_occupancy,
            "random_seed": self.random_seed,
            "initial_self_transition_probability": (
                self.initial_self_transition_probability
            ),
            "probability_pseudocount": self.probability_pseudocount,
            "maximum_per_observation_decrease": (self.maximum_per_observation_decrease),
            "initialization": "seeded_kmeans_plus_plus_n_init_1",
            "sequence_boundary_semantics": (
                "reset_initial_distribution_no_cross_sequence_transition"
            ),
            "restart_selection": (
                "highest_training_log_likelihood_then_lowest_restart_index"
            ),
            "student_t_degrees_of_freedom_rule": "fixed_not_estimated",
        }


@dataclass(frozen=True, slots=True)
class HMMRestartDiagnostic:
    """Auditable outcome for one deterministic restart."""

    restart_index: int
    seed: int
    converged: bool
    accepted: bool
    iterations: int
    training_log_likelihood: float | None
    final_per_observation_change: float | None = None
    state_occupancies: tuple[float, ...] = ()
    component_effective_counts: tuple[tuple[float, ...], ...] = ()
    failure_codes: tuple[HMMTrainingFailure, ...] = ()
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if type(self.restart_index) is not int or self.restart_index < 0:
            raise ValueError("restart_index must be a non-negative integer")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise ValueError("seed must be a 32-bit unsigned integer")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if self.training_log_likelihood is not None and not isfinite(
            self.training_log_likelihood
        ):
            raise ValueError("training_log_likelihood must be finite when present")
        if self.final_per_observation_change is not None and not isfinite(
            self.final_per_observation_change
        ):
            raise ValueError("final_per_observation_change must be finite when present")
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.state_occupancies
        ):
            raise ValueError("state occupancies must lie in [0, 1]")
        if self.state_occupancies and not np.isclose(
            sum(self.state_occupancies), 1.0, rtol=0.0, atol=1e-10
        ):
            raise ValueError("state occupancies must sum to one")
        if any(
            not isfinite(value) or value < 0.0
            for row in self.component_effective_counts
            for value in row
        ):
            raise ValueError("component effective counts must be non-negative")
        if self.accepted and (not self.converged or self.failure_codes):
            raise ValueError("an accepted restart must converge and pass every gate")


@dataclass(frozen=True, slots=True)
class HMMTrainingResult:
    """Selected immutable model plus a complete restart ledger."""

    model: FittedDiagonalHMM
    selected_restart_index: int
    restarts: tuple[HMMRestartDiagnostic, ...]
    training_data_sha256: str
    training_controls_sha256: str
    state_occupancies: tuple[float, ...]
    component_effective_counts: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, FittedDiagonalHMM):
            raise TypeError("model must be a FittedDiagonalHMM")
        if (
            len(self.training_data_sha256) != 64
            or len(self.training_controls_sha256) != 64
        ):
            raise ValueError("training hashes must be SHA-256 digests")
        selected = [
            item
            for item in self.restarts
            if item.restart_index == self.selected_restart_index
        ]
        if len(selected) != 1 or not selected[0].accepted:
            raise ValueError(
                "selected_restart_index must identify one accepted restart"
            )


class HMMTrainingError(RuntimeError):
    """Raised when all deterministic restarts fail their acceptance gates."""

    def __init__(
        self,
        message: str,
        *,
        restarts: tuple[HMMRestartDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.restarts = restarts


@dataclass(frozen=True, slots=True)
class DeterministicDiagonalHMMTrainer:
    """Shared NumPy EM trainer implementing the ``RegimeTrainer`` protocol."""

    controls: DiagonalHMMTrainingControls = field(
        default_factory=DiagonalHMMTrainingControls
    )

    def __post_init__(self) -> None:
        if not isinstance(self.controls, DiagonalHMMTrainingControls):
            raise TypeError("controls must be DiagonalHMMTrainingControls")

    def fit(
        self,
        batch: ObservationBatch,
        config: DiagonalHMMConfig,
        *,
        model_id: str,
        training_run_id: str | None = None,
        preprocessing_artifact_sha256: str | None = None,
    ) -> FittedDiagonalHMM:
        """Fit one declared model and return only the selected artifact."""

        return self.fit_with_diagnostics(
            batch,
            config,
            model_id=model_id,
            training_run_id=training_run_id,
            preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        ).model

    def fit_with_diagnostics(
        self,
        batch: ObservationBatch,
        config: DiagonalHMMConfig,
        *,
        model_id: str,
        training_run_id: str | None = None,
        preprocessing_artifact_sha256: str | None = None,
    ) -> HMMTrainingResult:
        """Fit a model and retain the complete deterministic restart ledger."""

        if not isinstance(batch, ObservationBatch):
            raise TypeError("batch must be an ObservationBatch")
        if not isinstance(config, DiagonalHMMConfig):
            raise TypeError("config must be a DiagonalHMMConfig")
        if type(model_id) is not str or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if training_run_id is not None and (
            type(training_run_id) is not str or not training_run_id.strip()
        ):
            raise ValueError("training_run_id must be a non-empty string when present")
        if preprocessing_artifact_sha256 is not None:
            _require_sha256(
                preprocessing_artifact_sha256,
                "preprocessing_artifact_sha256",
            )
        if batch.n_observations < config.n_states:
            raise ValueError("training observations must be at least the state count")
        if config.n_states * self.controls.minimum_state_occupancy > 1.0 + 1e-12:
            raise ValueError(
                "minimum_state_occupancy is infeasible for the requested state count"
            )

        prepared = _prepare_batch(batch)
        training_hash = hash_observation_batch(batch)
        controls_hash = _sha256_json(
            {
                "model_config": _config_state(config),
                "training_controls": self.controls.state_dict(),
            }
        )
        namespace = (
            f"{SHARED_HMM_TRAINER_VERSION}:{config.family.value}:"
            f"k={config.n_states}:m={config.n_mixtures}:"
            f"schema={batch.feature_schema.schema_sha256}:data={training_hash}"
        )
        seeds = _restart_seeds(
            self.controls.random_seed,
            self.controls.n_restarts,
            namespace=namespace,
        )
        diagnostics: list[HMMRestartDiagnostic] = []
        candidates: list[
            tuple[float, int, int, _RestartFit, tuple[float, ...], np.ndarray]
        ] = []

        for restart_index, seed in enumerate(seeds):
            try:
                restart = _fit_one_restart(
                    prepared,
                    config=config,
                    controls=self.controls,
                    seed=seed,
                )
                occupancy = tuple(
                    float(value) for value in restart.expectation.gamma.mean(axis=0)
                )
                counts = restart.expectation.component_gamma.sum(axis=0)
                failures: list[HMMTrainingFailure] = []
                if not restart.converged:
                    failures.append(HMMTrainingFailure.NON_CONVERGENCE)
                if any(
                    value < self.controls.minimum_state_occupancy for value in occupancy
                ):
                    failures.append(HMMTrainingFailure.OCCUPANCY_BELOW_FLOOR)
                failure_codes = tuple(dict.fromkeys(failures))
                accepted = not failure_codes
                diagnostics.append(
                    HMMRestartDiagnostic(
                        restart_index=restart_index,
                        seed=seed,
                        converged=restart.converged,
                        accepted=accepted,
                        iterations=restart.iterations,
                        training_log_likelihood=(restart.expectation.log_likelihood),
                        final_per_observation_change=(
                            restart.final_per_observation_change
                        ),
                        state_occupancies=occupancy,
                        component_effective_counts=_tuple_2d(counts),
                        failure_codes=failure_codes,
                    )
                )
                if accepted:
                    candidates.append(
                        (
                            restart.expectation.log_likelihood,
                            -restart_index,
                            seed,
                            restart,
                            occupancy,
                            counts,
                        )
                    )
            except _NumericalFitError as exc:
                diagnostics.append(
                    HMMRestartDiagnostic(
                        restart_index=restart_index,
                        seed=seed,
                        converged=False,
                        accepted=False,
                        iterations=exc.iterations,
                        training_log_likelihood=exc.log_likelihood,
                        final_per_observation_change=exc.final_per_observation_change,
                        failure_codes=(HMMTrainingFailure.NUMERICAL_ERROR,),
                        failure_message=str(exc),
                    )
                )
            except (
                FloatingPointError,
                OverflowError,
                ValueError,
                np.linalg.LinAlgError,
            ) as exc:
                diagnostics.append(
                    HMMRestartDiagnostic(
                        restart_index=restart_index,
                        seed=seed,
                        converged=False,
                        accepted=False,
                        iterations=0,
                        training_log_likelihood=None,
                        failure_codes=(HMMTrainingFailure.NUMERICAL_ERROR,),
                        failure_message=f"{type(exc).__name__}: {exc}",
                    )
                )

        restart_ledger = tuple(diagnostics)
        if not candidates:
            raise HMMTrainingError(
                "no deterministic HMM restart converged and passed occupancy gates",
                restarts=restart_ledger,
            )

        (
            _,
            negative_restart_index,
            _,
            selected,
            occupancy,
            component_counts,
        ) = max(candidates, key=lambda item: (item[0], item[1]))
        selected_restart_index = -negative_restart_index
        parameters, canonical_counts = _canonicalize_components(
            selected.parameters,
            component_counts,
        )
        model = FittedDiagonalHMM(
            model_id=model_id,
            family=config.family,
            feature_schema=batch.feature_schema,
            initial_probabilities=parameters.initial,
            transition_matrix=parameters.transition,
            mixture_weights=parameters.mixture_weights,
            means=parameters.means,
            diagonal_scales=parameters.scales,
            student_t_degrees_of_freedom=(
                config.student_t_degrees_of_freedom
                if config.family is HMMFamily.STUDENT_T
                else None
            ),
            training_run_id=training_run_id,
            preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        )
        return HMMTrainingResult(
            model=model,
            selected_restart_index=selected_restart_index,
            restarts=restart_ledger,
            training_data_sha256=training_hash,
            training_controls_sha256=controls_hash,
            state_occupancies=occupancy,
            component_effective_counts=_tuple_2d(canonical_counts),
        )


def hash_observation_batch(batch: ObservationBatch) -> str:
    """Hash exact rows, sequence boundaries, times, values, and feature order."""

    if not isinstance(batch, ObservationBatch):
        raise TypeError("batch must be an ObservationBatch")
    payload = {
        "implementation_version": SHARED_HMM_TRAINER_VERSION,
        "feature_schema": batch.feature_schema.state_dict(),
        "sequences": [
            {
                "sequence_id": sequence.sequence_id,
                "entity_id": sequence.entity_id,
                "row_ids": list(sequence.row_ids),
                "observation_times": [
                    value.isoformat() for value in sequence.observation_times
                ],
                "values": sequence.values.tolist(),
            }
            for sequence in sorted(batch.sequences, key=lambda item: item.sequence_id)
        ],
    }
    return _sha256_json(payload)


@dataclass(slots=True)
class _Parameters:
    initial: np.ndarray
    transition: np.ndarray
    mixture_weights: np.ndarray
    means: np.ndarray
    scales: np.ndarray


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    values: tuple[np.ndarray, ...]
    all_values: np.ndarray


@dataclass(frozen=True, slots=True)
class _Expectation:
    log_likelihood: float
    gamma: np.ndarray
    component_gamma: np.ndarray
    start_counts: np.ndarray
    transition_counts: np.ndarray


@dataclass(frozen=True, slots=True)
class _RestartFit:
    parameters: _Parameters
    expectation: _Expectation
    iterations: int
    converged: bool
    final_per_observation_change: float | None


class _NumericalFitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        iterations: int = 0,
        log_likelihood: float | None = None,
        final_per_observation_change: float | None = None,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.log_likelihood = log_likelihood
        self.final_per_observation_change = final_per_observation_change


def _prepare_batch(batch: ObservationBatch) -> _PreparedBatch:
    sequences = tuple(sorted(batch.sequences, key=lambda item: item.sequence_id))
    values = tuple(
        np.asarray(sequence.values, dtype=np.float64) for sequence in sequences
    )
    all_values = np.concatenate(values, axis=0)
    if all_values.ndim != 2 or not np.isfinite(all_values).all():
        raise ValueError("HMM training matrix must be finite and two-dimensional")
    return _PreparedBatch(values=values, all_values=all_values)


def _fit_one_restart(
    prepared: _PreparedBatch,
    *,
    config: DiagonalHMMConfig,
    controls: DiagonalHMMTrainingControls,
    seed: int,
) -> _RestartFit:
    parameters = _initial_parameters(
        prepared.all_values,
        config=config,
        controls=controls,
        seed=seed,
    )
    previous_log_likelihood: float | None = None
    latest: _Expectation | None = None
    change: float | None = None

    for iteration in range(1, controls.max_iterations + 1):
        try:
            expectation = _expectation(prepared, parameters, config=config)
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise _NumericalFitError(
                f"{type(exc).__name__}: {exc}",
                iterations=iteration,
                log_likelihood=(None if latest is None else latest.log_likelihood),
                final_per_observation_change=change,
            ) from exc
        latest = expectation
        current = expectation.log_likelihood
        if previous_log_likelihood is not None:
            change = (current - previous_log_likelihood) / prepared.all_values.shape[0]
            if change < -controls.maximum_per_observation_decrease:
                raise _NumericalFitError(
                    "EM log likelihood decreased beyond numerical tolerance",
                    iterations=iteration,
                    log_likelihood=current,
                    final_per_observation_change=change,
                )
            if 0.0 <= change <= controls.tolerance_per_observation:
                return _RestartFit(
                    parameters=parameters,
                    expectation=expectation,
                    iterations=iteration,
                    converged=True,
                    final_per_observation_change=change,
                )
        if iteration == controls.max_iterations:
            break
        try:
            parameters = _maximization(
                prepared.all_values,
                expectation,
                previous=parameters,
                config=config,
                controls=controls,
            )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise _NumericalFitError(
                f"{type(exc).__name__}: {exc}",
                iterations=iteration,
                log_likelihood=current,
                final_per_observation_change=change,
            ) from exc
        previous_log_likelihood = current

    if latest is None:  # validation makes this unreachable
        raise _NumericalFitError("EM did not evaluate a likelihood")
    return _RestartFit(
        parameters=parameters,
        expectation=latest,
        iterations=controls.max_iterations,
        converged=False,
        final_per_observation_change=change,
    )


def _initial_parameters(
    values: np.ndarray,
    *,
    config: DiagonalHMMConfig,
    controls: DiagonalHMMTrainingControls,
    seed: int,
) -> _Parameters:
    rng = np.random.default_rng(seed)
    state_count = config.n_states
    mixture_count = config.n_mixtures
    dimension = values.shape[1]
    initial = np.full(state_count, 1.0 / state_count, dtype=np.float64)
    off_diagonal = (1.0 - controls.initial_self_transition_probability) / (
        state_count - 1
    )
    transition = np.full((state_count, state_count), off_diagonal, dtype=np.float64)
    np.fill_diagonal(transition, controls.initial_self_transition_probability)
    state_centers = _seeded_kmeans_centers(values, state_count, rng)
    global_variance = np.maximum(np.var(values, axis=0), config.covariance_floor)
    if config.family is HMMFamily.STUDENT_T:
        degrees = config.student_t_degrees_of_freedom
        if degrees is None:  # guarded by DiagonalHMMConfig
            raise RuntimeError("Student-t config has no degrees of freedom")
        global_scale = np.maximum(
            global_variance * ((degrees - 2.0) / degrees),
            config.covariance_floor,
        )
    else:
        global_scale = global_variance
    means = np.repeat(state_centers[:, None, :], mixture_count, axis=1)
    if mixture_count > 1:
        distances = np.sum(
            (values[:, None, :] - state_centers[None, :, :]) ** 2,
            axis=2,
        )
        provisional_states = np.argmin(distances, axis=1)
        for state_index in range(state_count):
            members = values[provisional_states == state_index]
            means[state_index] = _seeded_kmeans_centers(
                members if len(members) else values,
                mixture_count,
                rng,
            )
    scales = np.broadcast_to(
        global_scale, (state_count, mixture_count, dimension)
    ).copy()
    mixture_weights = np.full(
        (state_count, mixture_count), 1.0 / mixture_count, dtype=np.float64
    )
    return _Parameters(initial, transition, mixture_weights, means, scales)


def _seeded_kmeans_centers(
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_observations = values.shape[0]
    centers = np.empty((count, values.shape[1]), dtype=np.float64)
    chosen = [int(rng.integers(0, n_observations))]
    centers[0] = values[chosen[0]]
    closest_distance = np.sum((values - centers[0]) ** 2, axis=1)
    for center_index in range(1, count):
        total = float(closest_distance.sum())
        if total > 0.0 and isfinite(total):
            candidate = int(rng.choice(n_observations, p=closest_distance / total))
        else:
            available = np.setdiff1d(np.arange(n_observations), np.asarray(chosen))
            candidate = int(rng.choice(available)) if available.size else chosen[0]
        chosen.append(candidate)
        centers[center_index] = values[candidate]
        distance = np.sum((values - centers[center_index]) ** 2, axis=1)
        closest_distance = np.minimum(closest_distance, distance)

    for _ in range(25):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        updated = centers.copy()
        for center_index in range(count):
            members = values[labels == center_index]
            if len(members):
                updated[center_index] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                updated[center_index] = values[farthest]
        if np.allclose(updated, centers, rtol=0.0, atol=1e-10):
            centers = updated
            break
        centers = updated
    order = sorted(
        range(count), key=lambda index: tuple(float(x) for x in centers[index])
    )
    return centers[np.asarray(order)]


def _expectation(
    prepared: _PreparedBatch,
    parameters: _Parameters,
    *,
    config: DiagonalHMMConfig,
) -> _Expectation:
    state_count = config.n_states
    mixture_count = config.n_mixtures
    log_initial = np.log(parameters.initial)
    log_transition = np.log(parameters.transition)
    gammas: list[np.ndarray] = []
    component_gammas: list[np.ndarray] = []
    start_counts = np.zeros(state_count, dtype=np.float64)
    transition_counts = np.zeros((state_count, state_count), dtype=np.float64)
    total_log_likelihood = 0.0

    for values in prepared.values:
        state_logs, component_logs = _emission_log_probabilities(
            values, parameters, config=config
        )
        gamma, transitions, log_likelihood = _forward_backward(
            state_logs,
            log_initial=log_initial,
            log_transition=log_transition,
        )
        conditional_components = np.exp(component_logs - state_logs[:, :, None])
        component_gamma = gamma[:, :, None] * conditional_components
        if not np.isfinite(component_gamma).all():
            raise FloatingPointError("non-finite component responsibility")
        gammas.append(gamma)
        component_gammas.append(component_gamma)
        start_counts += gamma[0]
        transition_counts += transitions
        total_log_likelihood += log_likelihood

    gamma_all = np.concatenate(gammas, axis=0)
    component_all = np.concatenate(component_gammas, axis=0)
    if gamma_all.shape[1:] != (state_count,) or component_all.shape[1:] != (
        state_count,
        mixture_count,
    ):
        raise RuntimeError("unexpected posterior geometry")
    return _Expectation(
        log_likelihood=float(total_log_likelihood),
        gamma=gamma_all,
        component_gamma=component_all,
        start_counts=start_counts,
        transition_counts=transition_counts,
    )


def _forward_backward(
    log_emissions: np.ndarray,
    *,
    log_initial: np.ndarray,
    log_transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    length, state_count = log_emissions.shape
    alpha = np.empty((length, state_count), dtype=np.float64)
    beta = np.zeros((length, state_count), dtype=np.float64)
    alpha[0] = log_initial + log_emissions[0]
    for row in range(1, length):
        alpha[row] = log_emissions[row] + _logsumexp(
            alpha[row - 1, :, None] + log_transition,
            axis=0,
        )
    sequence_log_likelihood = float(_logsumexp(alpha[-1], axis=None))

    for row in range(length - 2, -1, -1):
        beta[row] = _logsumexp(
            log_transition + log_emissions[row + 1][None, :] + beta[row + 1][None, :],
            axis=1,
        )

    log_gamma = alpha + beta
    log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
    gamma = np.exp(log_gamma)
    transition_sum = np.zeros((state_count, state_count), dtype=np.float64)
    for row in range(length - 1):
        log_xi = (
            alpha[row, :, None]
            + log_transition
            + log_emissions[row + 1][None, :]
            + beta[row + 1][None, :]
        )
        log_xi -= _logsumexp(log_xi, axis=None)
        transition_sum += np.exp(log_xi)
    if (
        not isfinite(sequence_log_likelihood)
        or not np.isfinite(gamma).all()
        or not np.isfinite(transition_sum).all()
    ):
        raise FloatingPointError("forward-backward produced non-finite values")
    return gamma, transition_sum, sequence_log_likelihood


def _emission_log_probabilities(
    values: np.ndarray,
    parameters: _Parameters,
    *,
    config: DiagonalHMMConfig,
) -> tuple[np.ndarray, np.ndarray]:
    differences = values[:, None, None, :] - parameters.means[None, :, :, :]
    dimension = values.shape[1]
    if config.family is HMMFamily.STUDENT_T:
        degrees = config.student_t_degrees_of_freedom
        if degrees is None:
            raise RuntimeError("Student-t config has no degrees of freedom")
        squared_distance = np.sum(
            differences**2 / parameters.scales[None, :, :, :], axis=3
        )
        constant = (
            lgamma((degrees + dimension) / 2.0)
            - lgamma(degrees / 2.0)
            - 0.5 * dimension * np.log(degrees * np.pi)
        )
        component_density = (
            constant
            - 0.5 * np.log(parameters.scales).sum(axis=2)[None, :, :]
            - 0.5 * (degrees + dimension) * np.log1p(squared_distance / degrees)
        )
    else:
        component_density = -0.5 * (
            dimension * np.log(2.0 * np.pi)
            + np.log(parameters.scales).sum(axis=2)[None, :, :]
            + np.sum(differences**2 / parameters.scales[None, :, :, :], axis=3)
        )
    component_logs = component_density + np.log(parameters.mixture_weights)[None, :, :]
    state_logs = _logsumexp(component_logs, axis=2)
    if not np.isfinite(state_logs).all() or not np.isfinite(component_logs).all():
        raise FloatingPointError("emission calculation produced non-finite values")
    return state_logs, component_logs


def _maximization(
    values: np.ndarray,
    expectation: _Expectation,
    *,
    previous: _Parameters,
    config: DiagonalHMMConfig,
    controls: DiagonalHMMTrainingControls,
) -> _Parameters:
    pseudocount = controls.probability_pseudocount
    initial = _normalize_probabilities(
        expectation.start_counts + pseudocount,
        floor=config.probability_floor,
    )
    transition = _normalize_probability_rows(
        expectation.transition_counts + pseudocount,
        floor=config.probability_floor,
    )
    component_counts = expectation.component_gamma.sum(axis=0)
    mixture_weights = _normalize_probability_rows(
        component_counts + pseudocount,
        floor=config.probability_floor,
    )
    state_count, mixture_count = component_counts.shape
    dimension = values.shape[1]
    means = np.empty((state_count, mixture_count, dimension), dtype=np.float64)
    scales = np.empty_like(means)

    if config.family is HMMFamily.STUDENT_T:
        degrees = config.student_t_degrees_of_freedom
        if degrees is None:
            raise RuntimeError("Student-t config has no degrees of freedom")
        old_differences = values[:, None, :] - previous.means[:, 0, :][None, :, :]
        old_distance = np.sum(
            old_differences**2 / previous.scales[:, 0, :][None, :, :], axis=2
        )
        latent_scale = (degrees + dimension) / (degrees + old_distance)
        weighted = expectation.gamma * latent_scale
        weighted_counts = weighted.sum(axis=0)
        state_counts = expectation.gamma.sum(axis=0)
        if np.any(weighted_counts <= 0.0) or np.any(state_counts <= 0.0):
            raise FloatingPointError("empty Student-t state during maximization")
        means[:, 0, :] = (
            np.einsum("nk,nd->kd", weighted, values) / weighted_counts[:, None]
        )
        for state_index in range(state_count):
            residual = values - means[state_index, 0]
            scales[state_index, 0] = (
                np.sum(
                    expectation.gamma[:, state_index, None]
                    * latent_scale[:, state_index, None]
                    * residual**2,
                    axis=0,
                )
                / state_counts[state_index]
            )
    else:
        if np.any(component_counts <= 0.0):
            raise FloatingPointError("empty Gaussian component during maximization")
        means = (
            np.einsum("nkm,nd->kmd", expectation.component_gamma, values)
            / component_counts[:, :, None]
        )
        for state_index in range(state_count):
            for component_index in range(mixture_count):
                residual = values - means[state_index, component_index]
                scales[state_index, component_index] = (
                    np.sum(
                        expectation.component_gamma[
                            :, state_index, component_index, None
                        ]
                        * residual**2,
                        axis=0,
                    )
                    / component_counts[state_index, component_index]
                )
    scales = np.maximum(scales, config.covariance_floor)
    if not np.isfinite(means).all() or not np.isfinite(scales).all():
        raise FloatingPointError("maximization produced non-finite emissions")
    return _Parameters(initial, transition, mixture_weights, means, scales)


def _canonicalize_components(
    parameters: _Parameters,
    component_effective_counts: np.ndarray,
) -> tuple[_Parameters, np.ndarray]:
    weights = parameters.mixture_weights.copy()
    means = parameters.means.copy()
    scales = parameters.scales.copy()
    counts = np.asarray(component_effective_counts, dtype=np.float64).copy()
    for state_index in range(means.shape[0]):
        order = sorted(
            range(means.shape[1]),
            key=lambda component_index: (
                tuple(float(value) for value in means[state_index, component_index]),
                tuple(float(value) for value in scales[state_index, component_index]),
                float(weights[state_index, component_index]),
            ),
        )
        indices = np.asarray(order, dtype=int)
        weights[state_index] = weights[state_index, indices]
        means[state_index] = means[state_index, indices]
        scales[state_index] = scales[state_index, indices]
        counts[state_index] = counts[state_index, indices]
    return (
        _Parameters(
            parameters.initial.copy(),
            parameters.transition.copy(),
            weights,
            means,
            scales,
        ),
        counts,
    )


def _normalize_probabilities(values: np.ndarray, *, floor: float) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), floor)
    total = float(array.sum())
    if not isfinite(total) or total <= 0.0:
        raise FloatingPointError("cannot normalize probability vector")
    normalized = np.maximum(array / total, floor)
    return normalized / normalized.sum()


def _normalize_probability_rows(values: np.ndarray, *, floor: float) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), floor)
    totals = array.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise FloatingPointError("cannot normalize probability rows")
    normalized = np.maximum(array / totals, floor)
    return normalized / normalized.sum(axis=1, keepdims=True)


def _logsumexp(values: np.ndarray, *, axis: int | None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    if axis is None:
        return np.asarray(result.squeeze())
    return np.squeeze(result, axis=axis)


def _restart_seeds(
    base_seed: int,
    count: int,
    *,
    namespace: str,
) -> tuple[int, ...]:
    return tuple(
        _derive_seed(base_seed, namespace, restart_index)
        for restart_index in range(count)
    )


def _derive_seed(base_seed: int, *tokens: Any) -> int:
    payload = json.dumps(
        {"base_seed": base_seed, "tokens": tokens},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:4],
        byteorder="big",
        signed=False,
    )


def _config_state(config: DiagonalHMMConfig) -> dict[str, Any]:
    return {
        "family": config.family.value,
        "n_states": config.n_states,
        "n_mixtures": config.n_mixtures,
        "covariance_floor": config.covariance_floor,
        "probability_floor": config.probability_floor,
        "student_t_degrees_of_freedom": (config.student_t_degrees_of_freedom),
    }


def _tuple_2d(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "DeterministicDiagonalHMMTrainer",
    "DiagonalHMMTrainingControls",
    "HMMRestartDiagnostic",
    "HMMTrainingError",
    "HMMTrainingFailure",
    "HMMTrainingResult",
    "SHARED_HMM_TRAINER_VERSION",
    "hash_observation_batch",
]
