"""Named Gaussian-mixture hidden Markov model estimator."""

from __future__ import annotations

from dataclasses import dataclass

from ._named import _NamedDiagonalHMM
from .base import DiagonalHMMConfig, HMMFamily


@dataclass(frozen=True, slots=True)
class GMMHMM(_NamedDiagonalHMM):
    """Diagonal GMM-HMM backed by the shared deterministic EM engine."""

    n_mixtures: int = 2

    def _build_config(self) -> DiagonalHMMConfig:
        return DiagonalHMMConfig(
            family=HMMFamily.GAUSSIAN_MIXTURE,
            n_states=self.n_states,
            n_mixtures=self.n_mixtures,
            covariance_floor=self.covariance_floor,
            probability_floor=self.probability_floor,
        )


__all__ = ["GMMHMM"]
