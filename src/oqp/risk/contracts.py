"""Typed contracts for risk-limit definitions and evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath


class RiskCalculationStatus(str, Enum):
    """Implementation maturity of a risk calculation."""

    ACTIVE = "active"
    MIGRATION = "migration"
    PLANNED = "planned"


class RiskEnforcementMode(str, Enum):
    """Approved runtime authority of a risk control."""

    OBSERVE = "observe"
    WARN = "warn"
    BLOCK = "block"


class RiskLimitDirection(str, Enum):
    """Direction in which a metric becomes less acceptable."""

    MAX = "max"
    MIN = "min"


class LimitEvaluationState(str, Enum):
    """Outcome of evaluating one risk control."""

    PLANNED = "planned"
    UNAVAILABLE = "unavailable"
    OBSERVED = "observed"
    PASS = "pass"
    WARNING = "warning"
    BREACH = "breach"


@dataclass(frozen=True, slots=True)
class RiskLimitDefinition:
    """One validated risk control from the department catalog."""

    control_id: str
    category: str
    scope: str
    unit: str
    direction: RiskLimitDirection
    calculation_status: RiskCalculationStatus
    enforcement_mode: RiskEnforcementMode
    warning_threshold: float | None
    hard_threshold: float | None
    metric_source_path: PurePosixPath
    metric_source_symbol: str
    owner: str
    description: str

    @property
    def metric_source(self) -> str:
        """Return the catalog representation of the calculation source."""

        return f"{self.metric_source_path.as_posix()}:{self.metric_source_symbol}"


@dataclass(frozen=True, slots=True)
class RiskLimitCatalog:
    """Validated collection of risk controls."""

    schema_version: str
    catalog_owner: str
    reporting_currency: str
    controls: tuple[RiskLimitDefinition, ...]
    source_path: Path

    def entry(self, control_id: str) -> RiskLimitDefinition | None:
        """Return the exact catalog control id, if registered."""

        key = str(control_id).strip()
        return next(
            (control for control in self.controls if control.control_id == key),
            None,
        )


@dataclass(frozen=True, slots=True)
class RiskLimitEvaluation:
    """Storage-neutral result for one evaluated risk control."""

    control_id: str
    state: LimitEvaluationState
    enforcement_mode: RiskEnforcementMode
    value: float | None
    warning_threshold: float | None
    hard_threshold: float | None
    message: str

    @property
    def blocks_action(self) -> bool:
        """Return whether an approved hard control should fail closed."""

        return self.enforcement_mode is RiskEnforcementMode.BLOCK and self.state in {
            LimitEvaluationState.BREACH,
            LimitEvaluationState.UNAVAILABLE,
        }
