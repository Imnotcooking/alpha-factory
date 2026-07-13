"""Contracts for the neutral, lagged defensive risk throttle.

The throttle scales one frozen benchmark return stream.  Strategy selection,
trend/mean-reversion routing, and same-period position formation are prohibited
by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from .filtering import PROSPECTIVE_SEMANTICS, ProbabilitySemantics


STAGE_OWNER = 10


class CostScenario(str, Enum):
    ZERO = "zero_cost"
    BASE = "base_cost"
    STRESSED = "stressed_cost"


@dataclass(frozen=True)
class CostAssumption:
    scenario: CostScenario
    one_way_turnover_cost_bps: float

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, CostScenario):
            raise TypeError("scenario must be a CostScenario")
        if not isfinite(self.one_way_turnover_cost_bps) or self.one_way_turnover_cost_bps < 0.0:
            raise ValueError("one-way turnover cost must be finite and non-negative")
        if self.scenario is CostScenario.ZERO and self.one_way_turnover_cost_bps != 0.0:
            raise ValueError("the zero-cost scenario must have a zero cost rate")


@dataclass(frozen=True)
class RiskThrottleConfig:
    benchmark_return_stream_id: str
    annual_volatility_target: float
    minimum_gross_multiplier: float
    maximum_gross_multiplier: float
    cost_assumptions: tuple[CostAssumption, ...]
    decision_delay_periods: int = 1
    require_average_exposure_matching: bool = True
    average_exposure_tolerance: float = 1e-6
    strategy_routing_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.benchmark_return_stream_id:
            raise ValueError("benchmark_return_stream_id is required")
        if not isfinite(self.annual_volatility_target) or self.annual_volatility_target <= 0.0:
            raise ValueError("annual_volatility_target must be finite and positive")
        bounds = (self.minimum_gross_multiplier, self.maximum_gross_multiplier)
        if any(not isfinite(value) or value < 0.0 for value in bounds):
            raise ValueError("gross-multiplier bounds must be finite and non-negative")
        if self.minimum_gross_multiplier > self.maximum_gross_multiplier:
            raise ValueError("minimum gross multiplier cannot exceed maximum")
        if self.decision_delay_periods < 1:
            raise ValueError("risk decisions must be applied at least one period later")
        if not self.require_average_exposure_matching:
            raise ValueError("paper comparisons require average-exposure matching")
        if not self.cost_assumptions:
            raise ValueError("cost_assumptions must be non-empty")
        scenarios = [item.scenario for item in self.cost_assumptions]
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("cost scenarios must be unique")
        if CostScenario.ZERO not in scenarios:
            raise ValueError("the cost ladder must include a zero-cost scenario")
        if not isfinite(self.average_exposure_tolerance) or self.average_exposure_tolerance < 0.0:
            raise ValueError("average_exposure_tolerance must be finite and non-negative")
        if self.strategy_routing_enabled:
            raise ValueError("Paper 1 prohibits strategy routing")


@dataclass(frozen=True)
class RiskForecastRecord:
    row_id: str
    product_id: str
    model_id: str
    feature_set_id: str
    decision_date: date
    application_date: date
    predicted_risk: float
    forecast_artifact_hash: str
    state_probability_semantics: ProbabilitySemantics | None = None

    def __post_init__(self) -> None:
        if self.state_probability_semantics is not None and not isinstance(
            self.state_probability_semantics, ProbabilitySemantics
        ):
            raise TypeError("state_probability_semantics must be a ProbabilitySemantics")
        if not all(
            (
                self.row_id,
                self.product_id,
                self.model_id,
                self.feature_set_id,
                self.forecast_artifact_hash,
            )
        ):
            raise ValueError("risk-forecast provenance is required")
        if self.application_date <= self.decision_date:
            raise ValueError("risk forecasts must be applied strictly after decision_date")
        if not isfinite(self.predicted_risk) or self.predicted_risk <= 0.0:
            raise ValueError("predicted_risk must be finite and positive")
        if self.state_probability_semantics is not None and (
            self.state_probability_semantics not in PROSPECTIVE_SEMANTICS
        ):
            raise ValueError("smoothed probabilities cannot drive risk decisions")


@dataclass(frozen=True)
class ExposureDecision:
    row_id: str
    product_id: str
    model_id: str
    decision_date: date
    application_date: date
    gross_multiplier: float
    forecast_artifact_hash: str

    def __post_init__(self) -> None:
        if not all((self.row_id, self.product_id, self.model_id, self.forecast_artifact_hash)):
            raise ValueError("exposure-decision provenance is required")
        if self.application_date <= self.decision_date:
            raise ValueError("an exposure decision cannot apply in its formation period")
        if not isfinite(self.gross_multiplier) or self.gross_multiplier < 0.0:
            raise ValueError("gross_multiplier must be finite and non-negative")


@dataclass(frozen=True)
class ExposureDecisionBatch:
    run_id: str
    config: RiskThrottleConfig
    decisions: tuple[ExposureDecision, ...]

    def __post_init__(self) -> None:
        if not self.run_id or not self.decisions:
            raise ValueError("run_id and exposure decisions are required")
        keys: set[tuple[str, date, str]] = set()
        for decision in self.decisions:
            if not (
                self.config.minimum_gross_multiplier
                <= decision.gross_multiplier
                <= self.config.maximum_gross_multiplier
            ):
                raise ValueError("gross multiplier violates configured bounds")
            key = (decision.product_id, decision.application_date, decision.model_id)
            if key in keys:
                raise ValueError("duplicate exposure decision")
            keys.add(key)


@dataclass(frozen=True)
class ThrottleEvaluationSummary:
    run_id: str
    model_id: str
    cost_scenario: CostScenario
    observations: int
    average_gross_exposure: float
    reference_average_gross_exposure: float
    average_exposure_tolerance: float
    net_certainty_equivalent: float
    tail_loss_frequency: float
    maximum_drawdown: float
    evaluation_artifact_hash: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.model_id or not self.evaluation_artifact_hash:
            raise ValueError("throttle evaluation provenance is required")
        if self.observations < 1:
            raise ValueError("throttle evaluation requires observations")
        values = (
            self.average_gross_exposure,
            self.reference_average_gross_exposure,
            self.average_exposure_tolerance,
            self.net_certainty_equivalent,
            self.tail_loss_frequency,
            self.maximum_drawdown,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("throttle summary values must be finite")
        if self.average_gross_exposure < 0.0 or self.reference_average_gross_exposure < 0.0:
            raise ValueError("average gross exposure cannot be negative")
        if self.average_exposure_tolerance < 0.0:
            raise ValueError("average_exposure_tolerance cannot be negative")
        if not 0.0 <= self.tail_loss_frequency <= 1.0:
            raise ValueError("tail_loss_frequency must lie in [0, 1]")
        if self.maximum_drawdown > 0.0:
            raise ValueError("maximum_drawdown uses the non-positive return convention")

    @property
    def exposure_matched(self) -> bool:
        return (
            abs(self.average_gross_exposure - self.reference_average_gross_exposure)
            <= self.average_exposure_tolerance
        )


@runtime_checkable
class RiskThrottle(Protocol):
    def build_decisions(
        self,
        forecasts: tuple[RiskForecastRecord, ...],
        config: RiskThrottleConfig,
    ) -> ExposureDecisionBatch:
        ...


@runtime_checkable
class RiskThrottleEvaluator(Protocol):
    def evaluate(
        self,
        decisions: ExposureDecisionBatch,
        benchmark_returns: object,
    ) -> tuple[ThrottleEvaluationSummary, ...]:
        ...
