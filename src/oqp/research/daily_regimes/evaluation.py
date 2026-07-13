"""Prospective evaluation contracts for the daily-regime paper.

Metric implementations are deliberately absent.  These types lock the common
panel, forecast-origin timing, probability semantics, and provenance that an
evaluation backend must honor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from math import isfinite
from typing import Protocol, runtime_checkable

from .filtering import PROSPECTIVE_SEMANTICS, ProbabilitySemantics


STAGE_OWNER = 8


class EvaluationSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


class EvaluationMetric(str, Enum):
    QLIKE = "qlike"
    BRIER_SCORE = "brier_score"
    CALIBRATION_ERROR = "calibration_error"
    ONE_STEP_LOG_PREDICTIVE_DENSITY = "one_step_log_predictive_density"
    ABSOLUTE_ERROR = "absolute_error"
    CERTAINTY_EQUIVALENT = "certainty_equivalent"
    TAIL_LOSS_FREQUENCY = "tail_loss_frequency"


class PredictionKind(str, Enum):
    POSITIVE_VARIANCE = "positive_variance"
    EVENT_PROBABILITY = "event_probability"
    LOG_PREDICTIVE_DENSITY = "log_predictive_density"
    CONTINUOUS = "continuous"


class TargetKind(str, Enum):
    NEXT_DAY_GK_GAP_VARIANCE = "next_day_gk_gap_variance"
    NEXT_DAY_TAIL_LOSS_EVENT = "next_day_tail_loss_event"
    NEXT_DAY_ABSOLUTE_RETURN = "next_day_absolute_return"
    FIVE_DAY_MAXIMUM_ADVERSE_MOVE = "five_day_maximum_adverse_move"


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[EvaluationMetric, ...]
    block_length_periods: int
    confidence_level: float = 0.95
    block_bootstrap: bool = True
    common_panel_required: bool = True
    paired_comparisons_required: bool = True
    filtered_probabilities_only: bool = True
    minimum_observations: int = 1

    def __post_init__(self) -> None:
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be non-empty and unique")
        if any(not isinstance(metric, EvaluationMetric) for metric in self.metrics):
            raise TypeError("metrics must contain EvaluationMetric values")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if self.block_bootstrap and self.block_length_periods < 1:
            raise ValueError("block bootstrap requires a positive frozen block length")
        if not self.block_bootstrap and self.block_length_periods < 0:
            raise ValueError("block_length_periods cannot be negative")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if not self.filtered_probabilities_only:
            raise ValueError("prospective evaluation cannot enable smoothed probabilities")


@dataclass(frozen=True)
class PredictionRecord:
    row_id: str
    product_id: str
    fold_id: str
    split: EvaluationSplit
    model_id: str
    feature_set_id: str
    target_kind: TargetKind
    prediction_kind: PredictionKind
    forecast_origin: date
    target_start_date: date
    target_end_date: date
    prediction: float
    observed: float
    refit_id: str
    prediction_artifact_hash: str
    state_probability_semantics: ProbabilitySemantics | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.split, EvaluationSplit):
            raise TypeError("split must be an EvaluationSplit")
        if not isinstance(self.target_kind, TargetKind):
            raise TypeError("target_kind must be a TargetKind")
        if not isinstance(self.prediction_kind, PredictionKind):
            raise TypeError("prediction_kind must be a PredictionKind")
        if self.state_probability_semantics is not None and not isinstance(
            self.state_probability_semantics, ProbabilitySemantics
        ):
            raise TypeError("state_probability_semantics must be a ProbabilitySemantics")
        required = (
            self.row_id,
            self.product_id,
            self.fold_id,
            self.model_id,
            self.feature_set_id,
            self.refit_id,
            self.prediction_artifact_hash,
        )
        if not all(required):
            raise ValueError("prediction provenance fields are required")
        if self.target_start_date <= self.forecast_origin:
            raise ValueError("every prospective target must begin after forecast_origin")
        if self.target_end_date < self.target_start_date:
            raise ValueError("target_end_date cannot precede target_start_date")
        if not isfinite(self.prediction) or not isfinite(self.observed):
            raise ValueError("prediction and observed values must be finite")
        if self.prediction_kind is PredictionKind.POSITIVE_VARIANCE:
            if self.prediction <= 0.0 or self.observed <= 0.0:
                raise ValueError("QLIKE inputs must be strictly positive")
        elif self.prediction_kind is PredictionKind.EVENT_PROBABILITY:
            if not 0.0 <= self.prediction <= 1.0 or self.observed not in {0.0, 1.0}:
                raise ValueError("event predictions require p in [0,1] and a binary outcome")
        if self.state_probability_semantics is not None and (
            self.state_probability_semantics not in PROSPECTIVE_SEMANTICS
        ):
            raise ValueError("smoothed state probabilities are forbidden in prediction records")


@dataclass(frozen=True)
class EvaluationPanel:
    panel_id: str
    common_panel_hash: str
    records: tuple[PredictionRecord, ...]

    def __post_init__(self) -> None:
        if not self.panel_id or not self.common_panel_hash or not self.records:
            raise ValueError("panel_id, common_panel_hash, and records are required")
        keys: set[tuple[str, str, TargetKind]] = set()
        for record in self.records:
            key = (record.row_id, record.model_id, record.target_kind)
            if key in keys:
                raise ValueError("duplicate prediction record in evaluation panel")
            keys.add(key)


@dataclass(frozen=True)
class MetricResult:
    metric: EvaluationMetric
    model_id: str
    feature_set_id: str
    target_kind: TargetKind
    split: EvaluationSplit
    estimate: float
    n_observations: int
    common_panel_hash: str
    lower_confidence_bound: float | None = None
    upper_confidence_bound: float | None = None
    standard_error: float | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.metric, EvaluationMetric):
            raise TypeError("metric must be an EvaluationMetric")
        if not isinstance(self.target_kind, TargetKind):
            raise TypeError("target_kind must be a TargetKind")
        if not isinstance(self.split, EvaluationSplit):
            raise TypeError("split must be an EvaluationSplit")
        if not self.model_id or not self.feature_set_id or not self.common_panel_hash:
            raise ValueError("metric provenance fields are required")
        if not isfinite(self.estimate) or self.n_observations < 1:
            raise ValueError("metric estimate must be finite with at least one observation")
        optional_values = (
            self.lower_confidence_bound,
            self.upper_confidence_bound,
            self.standard_error,
        )
        if any(value is not None and not isfinite(value) for value in optional_values):
            raise ValueError("confidence values must be finite when present")
        if (
            self.lower_confidence_bound is not None
            and self.upper_confidence_bound is not None
            and self.lower_confidence_bound > self.upper_confidence_bound
        ):
            raise ValueError("lower confidence bound cannot exceed upper confidence bound")
        if self.standard_error is not None and self.standard_error < 0.0:
            raise ValueError("standard_error cannot be negative")
        keys = [key for key, _ in self.metadata]
        if len(set(keys)) != len(keys):
            raise ValueError("metric metadata keys must be unique")


@dataclass(frozen=True)
class EvaluationResult:
    run_id: str
    config: EvaluationConfig
    panel_id: str
    common_panel_hash: str
    metrics: tuple[MetricResult, ...]

    def __post_init__(self) -> None:
        if not self.run_id or not self.panel_id or not self.common_panel_hash:
            raise ValueError("evaluation result provenance is required")
        if not self.metrics:
            raise ValueError("evaluation result must contain metrics")
        if any(metric.common_panel_hash != self.common_panel_hash for metric in self.metrics):
            raise ValueError("all metrics must use the declared common panel")


@runtime_checkable
class EvaluationBackend(Protocol):
    def score(self, panel: EvaluationPanel, config: EvaluationConfig) -> EvaluationResult:
        ...
