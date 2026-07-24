"""Canonical shared contracts for diagonal hidden Markov regime models.

This module intentionally defines interfaces rather than an estimator.  A
research trainer may use EM, Bayesian inference, or an external backend, while
every consumer receives the same immutable fitted-model and observation
contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Protocol, runtime_checkable


class HMMFamily(str, Enum):
    """Supported diagonal-emission HMM families."""

    GAUSSIAN = "gaussian_hmm"
    GAUSSIAN_MIXTURE = "gmm_hmm"
    STUDENT_T = "student_t_hmm"


@dataclass(frozen=True, slots=True)
class DiagonalHMMConfig:
    """Estimator-independent geometry and numerical controls.

    The object describes one concrete fit, not a hyperparameter search.  It is
    deliberately small so a training backend can add its own restart and
    convergence policy without changing the fitted-model contract.
    """

    family: HMMFamily
    n_states: int
    n_mixtures: int = 1
    covariance_floor: float = 1e-6
    probability_floor: float = 1e-12
    student_t_degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.family, HMMFamily):
            raise TypeError("family must be an HMMFamily")
        if type(self.n_states) is not int or self.n_states < 2:
            raise ValueError("n_states must be an integer of at least two")
        if type(self.n_mixtures) is not int or self.n_mixtures < 1:
            raise ValueError("n_mixtures must be a positive integer")
        if self.family is HMMFamily.GAUSSIAN_MIXTURE:
            if self.n_mixtures < 2:
                raise ValueError("a GMM-HMM requires at least two mixtures")
        elif self.n_mixtures != 1:
            raise ValueError("non-mixture HMM families require n_mixtures=1")
        if not isfinite(self.covariance_floor) or self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be finite and positive")
        if (
            not isfinite(self.probability_floor)
            or not 0.0 < self.probability_floor < 1.0
        ):
            raise ValueError("probability_floor must lie strictly in (0, 1)")
        if self.family is HMMFamily.STUDENT_T:
            degrees = self.student_t_degrees_of_freedom
            if degrees is None or not isfinite(degrees) or degrees <= 2.0:
                raise ValueError(
                    "Student-t HMMs require finite degrees of freedom above two"
                )
        elif self.student_t_degrees_of_freedom is not None:
            raise ValueError("only Student-t HMMs accept degrees of freedom")


@runtime_checkable
class RegimeTrainer(Protocol):
    """Offline fitting interface shared by interchangeable HMM backends."""

    def fit(
        self,
        batch: "ObservationBatch",
        config: DiagonalHMMConfig,
        *,
        model_id: str,
        training_run_id: str | None = None,
        preprocessing_artifact_sha256: str | None = None,
    ) -> "FittedDiagonalHMM":
        """Fit one declared configuration without mutating ``batch``."""
        ...


if TYPE_CHECKING:  # pragma: no cover
    from .fitted import FittedDiagonalHMM
    from .observations import ObservationBatch


__all__ = ["DiagonalHMMConfig", "HMMFamily", "RegimeTrainer"]
