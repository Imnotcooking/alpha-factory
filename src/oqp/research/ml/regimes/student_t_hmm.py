"""Named Student-t hidden Markov model estimator."""

from __future__ import annotations

from dataclasses import dataclass

from ._named import _NamedDiagonalHMM
from .base import DiagonalHMMConfig, HMMFamily


@dataclass(frozen=True, slots=True)
class StudentTHMM(_NamedDiagonalHMM):
    """Heavy-tailed diagonal HMM with fixed Student-t degrees of freedom."""

    degrees_of_freedom: float = 8.0

    def _build_config(self) -> DiagonalHMMConfig:
        return DiagonalHMMConfig(
            family=HMMFamily.STUDENT_T,
            n_states=self.n_states,
            covariance_floor=self.covariance_floor,
            probability_floor=self.probability_floor,
            student_t_degrees_of_freedom=self.degrees_of_freedom,
        )


__all__ = ["StudentTHMM"]
