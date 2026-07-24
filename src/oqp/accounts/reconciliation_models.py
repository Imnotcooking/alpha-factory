"""Typed, storage-neutral contracts for account reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite


class ReconciliationStatus(str, Enum):
    PASS = "pass"
    BREAK = "break"


class BreakCategory(str, Enum):
    IDENTITY = "identity"
    POSITION = "position"
    CASH = "cash"
    NAV = "nav"
    EVENT = "event"


class BreakSeverity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class NumericTolerance:
    """Absolute-or-relative tolerance for one numeric comparison."""

    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"NumericTolerance.{name} must be finite and non-negative")

    def threshold(self, reference: float) -> float:
        return max(float(self.absolute), abs(float(reference)) * float(self.relative))

    def permits(self, reference: float, observed: float) -> bool:
        if not isfinite(float(reference)) or not isfinite(float(observed)):
            return False
        return abs(float(observed) - float(reference)) <= self.threshold(reference)


@dataclass(frozen=True, slots=True)
class ReconciliationPolicy:
    """Explicit comparison scope and tolerances for one reconciliation run."""

    quantity: NumericTolerance = field(default_factory=NumericTolerance)
    multiplier: NumericTolerance = field(default_factory=NumericTolerance)
    market_value: NumericTolerance = field(default_factory=NumericTolerance)
    cash: NumericTolerance = field(default_factory=NumericTolerance)
    nav: NumericTolerance = field(default_factory=NumericTolerance)
    compare_market_value: bool = True
    compare_cash: bool = True
    compare_nav: bool = True
    allow_additional_observed_positions: bool = False
    max_snapshot_time_delta_seconds: float | None = None

    def __post_init__(self) -> None:
        value = self.max_snapshot_time_delta_seconds
        if value is not None and (not isfinite(float(value)) or float(value) < 0):
            raise ValueError(
                "ReconciliationPolicy.max_snapshot_time_delta_seconds must be "
                "finite and non-negative"
            )


ReconciliationValue = float | str | None


@dataclass(frozen=True, slots=True)
class ReconciliationBreak:
    category: BreakCategory
    severity: BreakSeverity
    key: str
    field: str
    reference_value: ReconciliationValue
    observed_value: ReconciliationValue
    difference: float | None
    tolerance: float | None
    message: str


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    run_id: str
    compared_at: datetime
    reference_snapshot_id: str
    observed_snapshot_id: str
    reference_label: str
    observed_label: str
    checks_performed: int
    breaks: tuple[ReconciliationBreak, ...]

    @property
    def status(self) -> ReconciliationStatus:
        return (
            ReconciliationStatus.BREAK
            if self.breaks
            else ReconciliationStatus.PASS
        )

    @property
    def break_count(self) -> int:
        return len(self.breaks)

    @property
    def critical_break_count(self) -> int:
        return sum(
            item.severity is BreakSeverity.CRITICAL for item in self.breaks
        )
