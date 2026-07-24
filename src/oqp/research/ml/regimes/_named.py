"""Private delegation shared by the named diagonal-HMM estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .base import DiagonalHMMConfig
from .trainer import (
    DeterministicDiagonalHMMTrainer,
    DiagonalHMMTrainingControls,
    HMMTrainingResult,
)


@dataclass(frozen=True, slots=True)
class _NamedDiagonalHMM:
    """A family-specific facade over the one deterministic EM implementation.

    The public subclasses own only configuration.  All fitting, restart,
    acceptance, serialization, and hashing semantics remain in
    :class:`DeterministicDiagonalHMMTrainer`.
    """

    n_states: int
    covariance_floor: float = 1e-6
    probability_floor: float = 1e-12
    controls: DiagonalHMMTrainingControls = field(
        default_factory=DiagonalHMMTrainingControls
    )

    def __post_init__(self) -> None:
        if not isinstance(self.controls, DiagonalHMMTrainingControls):
            raise TypeError("controls must be DiagonalHMMTrainingControls")
        # Constructing the immutable config applies the shared validation at
        # estimator construction time, not only once fitting has started.
        self._build_config()

    @property
    def config(self) -> DiagonalHMMConfig:
        """Return the exact low-level configuration delegated to the backend."""

        return self._build_config()

    def fit(
        self,
        batch: "ObservationBatch",
        *,
        model_id: str,
        training_run_id: str | None = None,
        preprocessing_artifact_sha256: str | None = None,
    ) -> "FittedDiagonalHMM":
        """Fit this declared family and return the selected immutable model."""

        return self._backend().fit(
            batch,
            self.config,
            model_id=model_id,
            training_run_id=training_run_id,
            preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        )

    def fit_with_diagnostics(
        self,
        batch: "ObservationBatch",
        *,
        model_id: str,
        training_run_id: str | None = None,
        preprocessing_artifact_sha256: str | None = None,
    ) -> HMMTrainingResult:
        """Fit this family and retain the deterministic restart ledger."""

        return self._backend().fit_with_diagnostics(
            batch,
            self.config,
            model_id=model_id,
            training_run_id=training_run_id,
            preprocessing_artifact_sha256=preprocessing_artifact_sha256,
        )

    def _backend(self) -> DeterministicDiagonalHMMTrainer:
        return DeterministicDiagonalHMMTrainer(controls=self.controls)

    def _build_config(self) -> DiagonalHMMConfig:
        raise NotImplementedError


if TYPE_CHECKING:  # pragma: no cover
    from .fitted import FittedDiagonalHMM
    from .observations import ObservationBatch


__all__: list[str] = []
