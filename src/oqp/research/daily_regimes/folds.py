"""Calendar-fold contracts for leakage-aware daily-regime evaluation.

Stage 5 owns the actual calendar cutoffs, target-dependent purge/embargo
arithmetic, and row assignment.  Stage 2 defines an auditable representation
and structural invariants for an expanding walk-forward plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd


STAGE_OWNER = 5
EXPANDING_MODE = "expanding_calendar_walk_forward"


class FoldConstructionUnavailableError(NotImplementedError):
    """Raised when scientific fold construction is requested before Stage 5."""


@dataclass(frozen=True)
class FoldPlanConfig:
    """Geometry requested from the Stage 5 calendar planner."""

    initial_training_periods: int
    validation_periods: int
    step_periods: int
    mode: str = EXPANDING_MODE
    purge_horizon_periods: int = 1
    embargo_periods: int = 0
    maximum_folds: int | None = None

    def __post_init__(self) -> None:
        if self.mode != EXPANDING_MODE:
            raise ValueError(f"Fold mode must be {EXPANDING_MODE!r}.")
        for name in (
            "initial_training_periods",
            "validation_periods",
            "step_periods",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one.")
        if self.purge_horizon_periods < 0:
            raise ValueError("purge_horizon_periods cannot be negative.")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods cannot be negative.")
        if self.maximum_folds is not None and self.maximum_folds < 1:
            raise ValueError("maximum_folds must be positive when supplied.")


@dataclass(frozen=True)
class CalendarFold:
    """One expanding training interval followed by one validation interval."""

    fold_id: str
    training_start: date
    training_end: date
    validation_start: date
    validation_end: date
    purge_horizon_periods: int
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id must be non-empty.")
        if self.training_end < self.training_start:
            raise ValueError("training_end cannot precede training_start.")
        if self.validation_end < self.validation_start:
            raise ValueError("validation_end cannot precede validation_start.")
        if self.training_end >= self.validation_start:
            raise ValueError("Training must end before validation begins.")
        if self.purge_horizon_periods < 0 or self.embargo_periods < 0:
            raise ValueError("Purge and embargo periods cannot be negative.")


@dataclass(frozen=True)
class FoldPlan:
    """Ordered immutable fold geometry plus planner diagnostics."""

    folds: tuple[CalendarFold, ...]
    planner_id: str
    mode: str = EXPANDING_MODE
    product_column: str = "product"
    trading_date_column: str = "trading_date"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.folds:
            raise ValueError("FoldPlan must contain at least one fold.")
        if not self.planner_id.strip():
            raise ValueError("planner_id must be non-empty.")
        if self.mode != EXPANDING_MODE:
            raise ValueError(f"FoldPlan.mode must be {EXPANDING_MODE!r}.")


@runtime_checkable
class FoldPlanner(Protocol):
    """Protocol for the target-aware planner introduced in Stage 5."""

    @property
    def planner_id(self) -> str:
        """Stable implementation identifier recorded in manifests."""

    def build(self, frame: pd.DataFrame, *, config: FoldPlanConfig) -> FoldPlan:
        """Construct folds from a calendar without inspecting target values."""


def validate_fold_plan(plan: FoldPlan) -> None:
    """Assert expanding chronology independently of row-assignment logic."""

    fold_ids = [fold.fold_id for fold in plan.folds]
    if len(fold_ids) != len(set(fold_ids)):
        raise ValueError("Fold identifiers must be unique.")

    first_training_start = plan.folds[0].training_start
    prior: CalendarFold | None = None
    for fold in plan.folds:
        if fold.training_start != first_training_start:
            raise ValueError("Expanding folds must retain the initial training start.")
        if prior is not None:
            if fold.training_end <= prior.training_end:
                raise ValueError("Training endpoints must increase across folds.")
            if fold.validation_start <= prior.validation_end:
                raise ValueError("Validation blocks must be chronological and disjoint.")
        prior = fold


def build_expanding_calendar_folds(
    frame: pd.DataFrame,
    *,
    config: FoldPlanConfig,
) -> FoldPlan:
    """Reserved entry point for Stage 5 target-aware fold construction."""

    del frame, config
    raise FoldConstructionUnavailableError(
        "Calendar cutoffs, target-dependent purges, and embargo row assignment "
        "are implemented in Stage 5.  Stage 2 provides interfaces only."
    )


__all__ = [
    "CalendarFold",
    "EXPANDING_MODE",
    "FoldConstructionUnavailableError",
    "FoldPlan",
    "FoldPlanConfig",
    "FoldPlanner",
    "STAGE_OWNER",
    "build_expanding_calendar_folds",
    "validate_fold_plan",
]
