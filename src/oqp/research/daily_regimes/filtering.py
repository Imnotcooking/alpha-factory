"""Prospective probability contracts for daily latent-state models.

The probability semantics are part of every record.  Smoothed probabilities
are deliberately representable for descriptive work, but the validation gates
in this module prevent them from entering prediction or decision interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import Protocol, TypeVar, runtime_checkable

from .contracts import (
    PROSPECTIVE_SEMANTICS,
    ProbabilitySemantics,
    ProbabilityUse,
    ProspectiveProbabilityError,
)
from .hmm import HMMFitResult, HMMSequence


STAGE_OWNER = 7


@dataclass(frozen=True)
class StateProbabilityRecord:
    product_id: str
    sequence_id: str
    row_id: str
    trading_date: date
    information_date: date
    model_id: str
    refit_id: str
    state_labels: tuple[str, ...]
    probabilities: tuple[float, ...]
    semantics: ProbabilitySemantics
    forecast_horizon_periods: int

    def __post_init__(self) -> None:
        if not isinstance(self.semantics, ProbabilitySemantics):
            raise TypeError("semantics must be a ProbabilitySemantics value")
        if not all((self.product_id, self.sequence_id, self.row_id, self.model_id, self.refit_id)):
            raise ValueError("probability record identifiers are required")
        if not self.state_labels or len(set(self.state_labels)) != len(self.state_labels):
            raise ValueError("state_labels must be non-empty and unique")
        if len(self.probabilities) != len(self.state_labels):
            raise ValueError("probability width must match state_labels")
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in self.probabilities):
            raise ValueError("state probabilities must be finite and lie in [0, 1]")
        if abs(sum(self.probabilities) - 1.0) > 1e-10:
            raise ValueError("state probabilities must sum to one")
        if self.semantics is ProbabilitySemantics.FILTERED:
            if self.information_date != self.trading_date:
                raise ValueError("filtered probabilities use information through their trading date")
            if self.forecast_horizon_periods != 0:
                raise ValueError("filtered probabilities must have horizon zero")
        elif self.semantics is ProbabilitySemantics.SMOOTHED:
            if self.information_date < self.trading_date:
                raise ValueError("smoothed information cannot predate the represented state")
            if self.forecast_horizon_periods != 0:
                raise ValueError("smoothed probabilities must have horizon zero")
        else:
            if self.information_date >= self.trading_date:
                raise ValueError("predicted states must be formed before their represented date")
            if self.forecast_horizon_periods != 1:
                raise ValueError("one-step probabilities require forecast horizon one")

    @property
    def prospective_eligible(self) -> bool:
        return self.semantics in PROSPECTIVE_SEMANTICS


@dataclass(frozen=True)
class StateProbabilityBatch:
    model_id: str
    feature_set_id: str
    fold_id: str
    records: tuple[StateProbabilityRecord, ...]

    def __post_init__(self) -> None:
        if not self.model_id or not self.feature_set_id or not self.fold_id:
            raise ValueError("model, feature-set, and fold identifiers are required")
        if not self.records:
            raise ValueError("a probability batch must contain records")
        keys: set[tuple[str, str, ProbabilitySemantics, int]] = set()
        width = len(self.records[0].state_labels)
        labels = self.records[0].state_labels
        for record in self.records:
            if record.model_id != self.model_id:
                raise ValueError("record model_id does not match batch model_id")
            if record.state_labels != labels or len(record.probabilities) != width:
                raise ValueError("state ordering must be identical throughout a batch")
            key = (
                record.sequence_id,
                record.row_id,
                record.semantics,
                record.forecast_horizon_periods,
            )
            if key in keys:
                raise ValueError("duplicate state-probability record")
            keys.add(key)

    def require_eligible_for(self, use: ProbabilityUse) -> None:
        if use is ProbabilityUse.DESCRIPTION:
            return
        invalid = [record.row_id for record in self.records if not record.prospective_eligible]
        if invalid:
            raise ProspectiveProbabilityError(
                f"{use.value} forbids smoothed probabilities; invalid rows: {invalid[:5]}"
            )


ModelT = TypeVar("ModelT")


@runtime_checkable
class ForwardFilter(Protocol[ModelT]):
    """Required online recursion; implementations may not call a smoother."""

    def filtered_probabilities(
        self,
        fitted: HMMFitResult[ModelT],
        sequence: HMMSequence,
        *,
        refit_id: str,
    ) -> StateProbabilityBatch:
        ...

    def one_step_probabilities(
        self,
        fitted: HMMFitResult[ModelT],
        filtered: StateProbabilityBatch,
        *,
        refit_id: str,
    ) -> StateProbabilityBatch:
        ...


@runtime_checkable
class HistoricalSmoother(Protocol[ModelT]):
    """Optional descriptive interface, intentionally separate from filtering."""

    def smoothed_probabilities(
        self,
        fitted: HMMFitResult[ModelT],
        sequence: HMMSequence,
        *,
        refit_id: str,
    ) -> StateProbabilityBatch:
        ...


__all__ = [
    "ForwardFilter",
    "HistoricalSmoother",
    "PROSPECTIVE_SEMANTICS",
    "ProbabilitySemantics",
    "ProbabilityUse",
    "ProspectiveProbabilityError",
    "STAGE_OWNER",
    "StateProbabilityBatch",
    "StateProbabilityRecord",
]
