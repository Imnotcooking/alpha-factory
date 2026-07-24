"""Named Gaussian hidden Markov model estimator."""

from __future__ import annotations

from dataclasses import dataclass

from ._named import _NamedDiagonalHMM
from .base import DiagonalHMMConfig, HMMFamily


@dataclass(frozen=True, slots=True)
class GaussianHMM(_NamedDiagonalHMM):
    """Diagonal Gaussian HMM backed by deterministic multi-restart EM."""

    def _build_config(self) -> DiagonalHMMConfig:
        return DiagonalHMMConfig(
            family=HMMFamily.GAUSSIAN,
            n_states=self.n_states,
            covariance_floor=self.covariance_floor,
            probability_floor=self.probability_floor,
        )


__all__ = ["GaussianHMM"]
