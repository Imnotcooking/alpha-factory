"""Canonical strategy-candidate contract for research-to-trading promotion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from oqp.domain import utc_now
from oqp.contracts.market_vertical import normalize_market_vertical


class CandidateStatus(str, Enum):
    RESEARCH_ONLY = "research_only"
    MARKET_REVIEW = "market_review"
    PAPER_CANDIDATE = "paper_candidate"
    PAPER_RUNNING = "paper_running"
    REJECTED = "rejected"
    RETIRED = "retired"
    OUT_OF_SCOPE = "out_of_scope"
    TRANSLATION_REQUIRED = "translation_required"  # Legacy alias for old artifacts.


class CandidateIntakeState(str, Enum):
    RESEARCH_SNAPSHOT = "research_snapshot"
    NEEDS_REVIEW = "needs_review"
    PAPER_QUEUE_ELIGIBLE = "paper_queue_eligible"
    PAPER_RUNNING = "paper_running"


_INTAKE_STATE_LABELS = {
    CandidateIntakeState.RESEARCH_SNAPSHOT: "Research Snapshot",
    CandidateIntakeState.NEEDS_REVIEW: "Needs Review",
    CandidateIntakeState.PAPER_QUEUE_ELIGIBLE: "Paper Queue Eligible",
    CandidateIntakeState.PAPER_RUNNING: "Paper Running",
}


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    validation_ic: float | None = None
    holdout_ic: float | None = None
    crisis_ic: float | None = None
    validation_hit_rate: float | None = None
    holdout_hit_rate: float | None = None
    sharpe_ratio: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    turnover_rate: float | None = None
    avg_daily_cost_bps: float | None = None
    metric_p_value: float | None = None
    sharpe_p_value: float | None = None
    significance: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSafetyLimits:
    paper_only: bool = True
    allow_live_trading: bool = False
    max_gross_exposure: float | None = None
    max_single_position: float | None = None
    max_daily_loss_pct: float | None = None
    max_order_notional: float | None = None

    def __post_init__(self) -> None:
        if self.paper_only and self.allow_live_trading:
            raise ValueError("paper_only candidates cannot allow live trading")
        for field_name in (
            "max_gross_exposure",
            "max_single_position",
            "max_daily_loss_pct",
            "max_order_notional",
        ):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    candidate_id: str
    strategy_id: str
    source: str
    promotion_status: CandidateStatus = CandidateStatus.RESEARCH_ONLY
    native_market_vertical: str = "UNKNOWN"
    tested_market_vertical: str | None = None
    target_market_vertical: str | None = None
    intended_market_verticals: tuple[str, ...] = ()
    research_run_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    dataset_id: str | None = None
    universe_id: str | None = None
    data_frequency: str | None = None
    data_vendor: str | None = None
    execution_assumption: str | None = None
    evaluation_geometry: str | None = None
    ic_metric: str | None = None
    metrics: CandidateMetrics = field(default_factory=CandidateMetrics)
    safety_limits: CandidateSafetyLimits = field(default_factory=CandidateSafetyLimits)
    instrument_mapping_required: bool = False
    approved_broker_profile: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("StrategyCandidate.candidate_id is required")
        if not self.strategy_id.strip():
            raise ValueError("StrategyCandidate.strategy_id is required")
        if not self.source.strip():
            raise ValueError("StrategyCandidate.source is required")

        object.__setattr__(
            self,
            "promotion_status",
            _candidate_status(self.promotion_status),
        )
        native = normalize_market_vertical(self.native_market_vertical)
        tested = normalize_market_vertical(self.tested_market_vertical or native)
        target = normalize_market_vertical(self.target_market_vertical or tested)
        intended = tuple(
            sorted(
                {
                    normalize_market_vertical(value)
                    for value in (self.intended_market_verticals or (native,))
                    if normalize_market_vertical(value) != "UNKNOWN"
                }
            )
        )
        object.__setattr__(self, "native_market_vertical", native)
        object.__setattr__(self, "tested_market_vertical", tested)
        object.__setattr__(self, "target_market_vertical", target)
        object.__setattr__(self, "intended_market_verticals", intended or (native,))

        if not isinstance(self.metrics, CandidateMetrics):
            object.__setattr__(self, "metrics", CandidateMetrics(**dict(self.metrics)))
        if not isinstance(self.safety_limits, CandidateSafetyLimits):
            object.__setattr__(
                self,
                "safety_limits",
                CandidateSafetyLimits(**dict(self.safety_limits)),
            )
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        if not isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def can_enter_paper_queue(self) -> bool:
        return (
            self.promotion_status == CandidateStatus.PAPER_CANDIDATE
            and self.safety_limits.paper_only
            and not self.safety_limits.allow_live_trading
            and self.target_market_vertical == self.tested_market_vertical
        )

    @property
    def intake_state(self) -> CandidateIntakeState:
        if self.can_enter_paper_queue:
            return CandidateIntakeState.PAPER_QUEUE_ELIGIBLE
        if self.promotion_status == CandidateStatus.PAPER_RUNNING:
            return CandidateIntakeState.PAPER_RUNNING
        if (
            self.promotion_status
            in {CandidateStatus.MARKET_REVIEW, CandidateStatus.PAPER_CANDIDATE}
            or self.is_cross_market_translation
            or self.instrument_mapping_required
        ):
            return CandidateIntakeState.NEEDS_REVIEW
        return CandidateIntakeState.RESEARCH_SNAPSHOT

    @property
    def intake_state_label(self) -> str:
        return _INTAKE_STATE_LABELS[self.intake_state]

    @property
    def is_cross_market_translation(self) -> bool:
        return self.target_market_vertical != self.tested_market_vertical

    @property
    def market_scoped_status(self) -> str:
        return f"{self.target_market_vertical}:{self.promotion_status.value}"


def _candidate_status(value: CandidateStatus | str) -> CandidateStatus:
    if isinstance(value, CandidateStatus):
        return value
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return CandidateStatus(normalized)
    except ValueError as exc:
        raise ValueError(f"Unknown candidate promotion status: {value!r}") from exc
