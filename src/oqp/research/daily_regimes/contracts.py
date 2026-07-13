"""Typed frame and fold contracts for daily latent-regime research."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


RAW_DAILY_BAR_REQUIRED_COLUMNS = (
    "product",
    "contract",
    "exchange",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "open_interest",
    "multiplier",
    "tick_size",
    "limit_lock_flag",
    "stale_bar_flag",
    "source_row_id",
    "listing_date",
    "last_trade_date",
)

RAW_DAILY_BAR_OPTIONAL_COLUMNS = (
    "is_dominant",
    "roll_flag",
    "quality_flag",
)


class ContractViolation(ValueError):
    """Raised when a frame violates a declared research contract."""


class ProbabilitySemantics(str, Enum):
    """Information set conditioning a state-probability vector."""

    FILTERED = "filtered_p_s_t_given_f_t"
    ONE_STEP_PREDICTED = "predicted_p_s_t_plus_1_given_f_t"
    SMOOTHED = "smoothed_p_s_t_given_f_T"


PROSPECTIVE_SEMANTICS = frozenset(
    {ProbabilitySemantics.FILTERED, ProbabilitySemantics.ONE_STEP_PREDICTED}
)


class ProbabilityUse(str, Enum):
    DESCRIPTION = "description"
    PROSPECTIVE_SCORE = "prospective_score"
    RISK_DECISION = "risk_decision"


class ProspectiveProbabilityError(ContractViolation):
    """Raised when a future-conditioned probability reaches prospective use."""


@dataclass(frozen=True, slots=True)
class FrameValidationReport:
    contract_name: str
    row_count: int
    column_count: int
    start_date: str | None
    end_date: str | None
    entity_count: int
    is_sorted: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RawDailyBarContract:
    """Point-in-time column and value contract for raw daily futures bars."""

    product_col: str = "product"
    contract_col: str = "contract"
    exchange_col: str = "exchange"
    date_col: str = "trading_date"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    turnover_col: str = "turnover"
    open_interest_col: str = "open_interest"
    multiplier_col: str = "multiplier"
    tick_size_col: str = "tick_size"
    limit_lock_col: str = "limit_lock_flag"
    stale_bar_col: str = "stale_bar_flag"
    source_row_id_col: str = "source_row_id"
    listing_date_col: str = "listing_date"
    last_trade_date_col: str = "last_trade_date"
    require_sorted: bool = False
    allow_empty: bool = False

    @property
    def required_columns(self) -> tuple[str, ...]:
        return (
            self.product_col,
            self.contract_col,
            self.exchange_col,
            self.date_col,
            self.open_col,
            self.high_col,
            self.low_col,
            self.close_col,
            self.volume_col,
            self.turnover_col,
            self.open_interest_col,
            self.multiplier_col,
            self.tick_size_col,
            self.limit_lock_col,
            self.stale_bar_col,
            self.source_row_id_col,
            self.listing_date_col,
            self.last_trade_date_col,
        )

    def validate(self, frame: pd.DataFrame) -> FrameValidationReport:
        _require_dataframe(frame, "raw daily bars")
        _require_columns(frame, self.required_columns, "raw daily bars")
        if frame.empty:
            if self.allow_empty:
                return FrameValidationReport(
                    contract_name="raw_daily_bars",
                    row_count=0,
                    column_count=len(frame.columns),
                    start_date=None,
                    end_date=None,
                    entity_count=0,
                    is_sorted=True,
                )
            raise ContractViolation("Raw daily bar frame cannot be empty.")

        dates = _valid_dates(frame[self.date_col], self.date_col)
        if not dates.equals(dates.dt.normalize()):
            raise ContractViolation("trading_date must contain normalized daily dates.")
        _require_nonempty_strings(frame[self.product_col], self.product_col)
        _require_nonempty_strings(frame[self.contract_col], self.contract_col)
        _require_nonempty_strings(frame[self.exchange_col], self.exchange_col)
        _require_nonempty_strings(frame[self.source_row_id_col], self.source_row_id_col)
        _require_unique(
            frame,
            (self.product_col, self.contract_col, self.date_col),
            "raw daily bars",
        )
        _require_unique(
            frame,
            (self.source_row_id_col,),
            "raw daily bar source rows",
        )

        prices = {
            name: _finite_numeric(frame[column], column)
            for name, column in (
                ("open", self.open_col),
                ("high", self.high_col),
                ("low", self.low_col),
                ("close", self.close_col),
            )
        }
        for name, values in prices.items():
            if (values <= 0).any():
                raise ContractViolation(f"{name} prices must be strictly positive.")
        if "settlement" in frame.columns:
            settlement = _finite_numeric(frame["settlement"], "settlement")
            if (settlement <= 0).any():
                raise ContractViolation(
                    "settlement prices must be strictly positive."
                )
        high_floor = np.maximum.reduce(
            [prices["open"].to_numpy(), prices["low"].to_numpy(), prices["close"].to_numpy()]
        )
        low_ceiling = np.minimum.reduce(
            [prices["open"].to_numpy(), prices["high"].to_numpy(), prices["close"].to_numpy()]
        )
        if (prices["high"].to_numpy() < high_floor).any():
            raise ContractViolation("high must be at least open, low, and close on every row.")
        if (prices["low"].to_numpy() > low_ceiling).any():
            raise ContractViolation("low must be at most open, high, and close on every row.")

        volume = _finite_numeric(frame[self.volume_col], self.volume_col)
        turnover = _finite_numeric(frame[self.turnover_col], self.turnover_col)
        open_interest = _finite_numeric(
            frame[self.open_interest_col], self.open_interest_col
        )
        multiplier = _finite_numeric(frame[self.multiplier_col], self.multiplier_col)
        tick_size = _finite_numeric(frame[self.tick_size_col], self.tick_size_col)
        if (volume < 0).any():
            raise ContractViolation("volume cannot be negative.")
        if (turnover < 0).any():
            raise ContractViolation("turnover cannot be negative.")
        if (open_interest < 0).any():
            raise ContractViolation("open_interest cannot be negative.")
        if (multiplier <= 0).any():
            raise ContractViolation("multiplier must be strictly positive.")
        if (tick_size <= 0).any():
            raise ContractViolation("tick_size must be strictly positive.")

        for boolean_col in (
            self.limit_lock_col,
            self.stale_bar_col,
            "is_dominant",
            "roll_flag",
        ):
            if boolean_col in frame.columns:
                _require_boolean_like(frame[boolean_col], boolean_col)
        if "quality_flag" in frame.columns:
            _require_nonempty_strings(frame["quality_flag"], "quality_flag")

        listing_dates: pd.Series | None = None
        last_trade_dates: pd.Series | None = None
        if self.listing_date_col in frame.columns:
            listing_dates = _valid_dates(
                frame[self.listing_date_col], self.listing_date_col
            )
            if (listing_dates.dt.normalize() > dates.dt.normalize()).any():
                raise ContractViolation("listing_date cannot follow trading_date.")
        if self.last_trade_date_col in frame.columns:
            last_trade_dates = _valid_dates(
                frame[self.last_trade_date_col], self.last_trade_date_col
            )
            if (last_trade_dates.dt.normalize() < dates.dt.normalize()).any():
                raise ContractViolation("last_trade_date cannot precede trading_date.")
        if listing_dates is not None and last_trade_dates is not None:
            if (
                listing_dates.dt.normalize() > last_trade_dates.dt.normalize()
            ).any():
                raise ContractViolation("listing_date cannot follow last_trade_date.")

        sort_columns = [self.product_col, self.contract_col, self.date_col]
        is_sorted = _is_sorted(frame, sort_columns)
        if self.require_sorted and not is_sorted:
            raise ContractViolation(
                f"Raw daily bars must be sorted by {sort_columns}."
            )
        warning_items: list[str] = []
        if (volume == 0).any():
            warning_items.append("zero_volume_rows_present")
        if ((volume == 0) & (turnover > 0)).any():
            warning_items.append("zero_volume_positive_turnover_rows_present")
        if (open_interest == 0).any():
            warning_items.append("zero_open_interest_rows_present")
        if frame[self.limit_lock_col].astype(bool).any():
            warning_items.append("limit_locked_rows_present")
        if frame[self.stale_bar_col].astype(bool).any():
            warning_items.append("stale_rows_present")
        return FrameValidationReport(
            contract_name="raw_daily_bars",
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            start_date=_date_string(dates.min()),
            end_date=_date_string(dates.max()),
            entity_count=int(frame[self.product_col].astype(str).nunique()),
            is_sorted=is_sorted,
            warnings=tuple(warning_items),
        )


@dataclass(frozen=True, slots=True)
class FeatureFrameContract:
    """Contract for decision-time feature matrices."""

    feature_columns: tuple[str, ...]
    product_col: str = "product"
    date_col: str = "trading_date"
    decision_time_col: str | None = None
    allow_missing_features: bool = False
    require_sorted: bool = False

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ContractViolation("FeatureFrameContract requires feature columns.")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ContractViolation("Feature columns cannot contain duplicates.")

    def validate(self, frame: pd.DataFrame) -> FrameValidationReport:
        _require_dataframe(frame, "feature frame")
        required = [self.product_col, self.date_col, *self.feature_columns]
        if self.decision_time_col:
            required.append(self.decision_time_col)
        _require_columns(frame, tuple(required), "feature frame")
        if frame.empty:
            raise ContractViolation("Feature frame cannot be empty.")
        dates = _valid_dates(frame[self.date_col], self.date_col)
        _require_nonempty_strings(frame[self.product_col], self.product_col)
        _require_unique(frame, (self.product_col, self.date_col), "feature frame")
        for column in self.feature_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if not self.allow_missing_features and numeric.isna().any():
                raise ContractViolation(f"Feature column {column!r} contains missing values.")
            finite = numeric.dropna().to_numpy(dtype=float)
            if finite.size and not np.isfinite(finite).all():
                raise ContractViolation(f"Feature column {column!r} contains non-finite values.")
        if self.decision_time_col:
            decision_times = _valid_dates(
                frame[self.decision_time_col], self.decision_time_col
            )
            if (decision_times.dt.normalize() < dates.dt.normalize()).any():
                raise ContractViolation(
                    "decision_time cannot precede its feature trading_date."
                )
        sort_columns = [self.product_col, self.date_col]
        is_sorted = _is_sorted(frame, sort_columns)
        if self.require_sorted and not is_sorted:
            raise ContractViolation(f"Feature frame must be sorted by {sort_columns}.")
        return FrameValidationReport(
            contract_name="feature_frame",
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            start_date=_date_string(dates.min()),
            end_date=_date_string(dates.max()),
            entity_count=int(frame[self.product_col].astype(str).nunique()),
            is_sorted=is_sorted,
        )


@dataclass(frozen=True, slots=True)
class StateProbabilityContract:
    """DataFrame contract with mandatory probability-information semantics.

    Prospective scoring is the safe default.  Historical smoothing must be
    requested explicitly through ``ProbabilityUse.DESCRIPTION`` and can never
    pass silently into prediction or risk-decision consumers.
    """

    probability_columns: tuple[str, ...] = field(default_factory=tuple)
    product_col: str = "product"
    date_col: str = "trading_date"
    information_date_col: str = "information_date"
    fold_col: str = "fold_id"
    model_col: str = "model_id"
    state_col: str = "state"
    semantics_col: str = "probability_semantics"
    forecast_horizon_col: str = "forecast_horizon_periods"
    intended_use: ProbabilityUse = ProbabilityUse.PROSPECTIVE_SCORE
    probability_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.intended_use, ProbabilityUse):
            raise ContractViolation("intended_use must be a ProbabilityUse value.")
        if self.probability_tolerance <= 0:
            raise ContractViolation("probability_tolerance must be positive.")

    def validate(self, frame: pd.DataFrame) -> FrameValidationReport:
        _require_dataframe(frame, "state probabilities")
        probability_columns = self.probability_columns or tuple(
            column for column in frame.columns if str(column).startswith("p_state_")
        )
        if len(probability_columns) < 2:
            raise ContractViolation("At least two state-probability columns are required.")
        required = (
            self.product_col,
            self.date_col,
            self.information_date_col,
            self.fold_col,
            self.model_col,
            self.semantics_col,
            self.forecast_horizon_col,
            *probability_columns,
        )
        _require_columns(frame, required, "state probabilities")
        if frame.empty:
            raise ContractViolation("State-probability frame cannot be empty.")
        dates = _valid_dates(frame[self.date_col], self.date_col)
        information_dates = _valid_dates(
            frame[self.information_date_col], self.information_date_col
        )
        try:
            semantics = pd.Series(
                [ProbabilitySemantics(value) for value in frame[self.semantics_col]],
                index=frame.index,
            )
        except (TypeError, ValueError) as exc:
            allowed = [value.value for value in ProbabilitySemantics]
            raise ContractViolation(
                f"{self.semantics_col} must contain one of {allowed}."
            ) from exc
        horizons = _finite_numeric(
            frame[self.forecast_horizon_col], self.forecast_horizon_col
        )
        if not np.equal(horizons, np.floor(horizons)).all() or (horizons < 0).any():
            raise ContractViolation(
                f"{self.forecast_horizon_col} must contain non-negative integers."
            )

        trading_days = dates.dt.normalize()
        information_days = information_dates.dt.normalize()
        filtered = semantics.eq(ProbabilitySemantics.FILTERED)
        predicted = semantics.eq(ProbabilitySemantics.ONE_STEP_PREDICTED)
        smoothed = semantics.eq(ProbabilitySemantics.SMOOTHED)
        if (filtered & (information_days != trading_days)).any() or (
            filtered & horizons.ne(0)
        ).any():
            raise ContractViolation(
                "Filtered probabilities require same-day information and horizon zero."
            )
        if (predicted & (information_days >= trading_days)).any() or (
            predicted & horizons.ne(1)
        ).any():
            raise ContractViolation(
                "One-step predictions require earlier information and horizon one."
            )
        if (smoothed & (information_days < trading_days)).any() or (
            smoothed & horizons.ne(0)
        ).any():
            raise ContractViolation(
                "Smoothed probabilities require information no earlier than the state "
                "date and horizon zero."
            )
        if self.intended_use is not ProbabilityUse.DESCRIPTION and smoothed.any():
            raise ProspectiveProbabilityError(
                f"{self.intended_use.value} forbids smoothed probabilities."
            )
        _require_unique(
            frame,
            (self.product_col, self.date_col, self.fold_col, self.model_col),
            "state probabilities",
        )
        probabilities = frame[list(probability_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        if probabilities.isna().any().any():
            raise ContractViolation("State probabilities cannot be missing or non-numeric.")
        values = probabilities.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ContractViolation("State probabilities must be finite.")
        if ((values < -self.probability_tolerance) | (values > 1 + self.probability_tolerance)).any():
            raise ContractViolation("State probabilities must lie in [0, 1].")
        if not np.allclose(values.sum(axis=1), 1.0, atol=self.probability_tolerance, rtol=0.0):
            raise ContractViolation("State probabilities must sum to one on every row.")
        if self.state_col in frame.columns:
            states = pd.to_numeric(frame[self.state_col], errors="coerce")
            if states.isna().any() or not np.equal(states, np.floor(states)).all():
                raise ContractViolation("state must contain integer state identifiers.")
            if ((states < 0) | (states >= len(probability_columns))).any():
                raise ContractViolation("state identifiers are outside the probability column range.")
        return FrameValidationReport(
            contract_name="state_probabilities",
            row_count=int(len(frame)),
            column_count=int(len(frame.columns)),
            start_date=_date_string(dates.min()),
            end_date=_date_string(dates.max()),
            entity_count=int(frame[self.product_col].astype(str).nunique()),
            is_sorted=_is_sorted(
                frame, [self.model_col, self.fold_col, self.product_col, self.date_col]
            ),
        )


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """Immutable expanding-window fold boundary."""

    fold_id: str
    train_start: str
    train_end: str
    evaluation_start: str
    evaluation_end: str
    purge_periods: int = 0
    embargo_periods: int = 0

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ContractViolation("fold_id cannot be empty.")
        if self.purge_periods < 0 or self.embargo_periods < 0:
            raise ContractViolation("Fold purge and embargo periods cannot be negative.")
        train_start = _timestamp(self.train_start, "train_start")
        train_end = _timestamp(self.train_end, "train_end")
        evaluation_start = _timestamp(self.evaluation_start, "evaluation_start")
        evaluation_end = _timestamp(self.evaluation_end, "evaluation_end")
        if train_end < train_start:
            raise ContractViolation("train_end cannot precede train_start.")
        if evaluation_end < evaluation_start:
            raise ContractViolation("evaluation_end cannot precede evaluation_start.")
        if evaluation_start <= train_end:
            raise ContractViolation(
                "Expanding-fold evaluation must begin strictly after the training window."
            )


@dataclass(frozen=True, slots=True)
class StageRunResult:
    """Small structured result shared by package-owned runner stages."""

    stage: str
    status: str
    row_count: int = 0
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, Any] = field(default_factory=dict)
    messages: tuple[str, ...] = field(default_factory=tuple)


def _require_dataframe(value: Any, context: str) -> None:
    if not isinstance(value, pd.DataFrame):
        raise ContractViolation(f"{context} must be a pandas DataFrame.")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractViolation(f"{context} is missing required columns: {missing}")


def _require_unique(frame: pd.DataFrame, columns: tuple[str, ...], context: str) -> None:
    duplicate_mask = frame.duplicated(list(columns), keep=False)
    if duplicate_mask.any():
        raise ContractViolation(
            f"{context} contains {int(duplicate_mask.sum())} rows with duplicate keys {columns}."
        )


def _valid_dates(series: pd.Series, name: str) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    if dates.isna().any():
        raise ContractViolation(f"{name} contains invalid or missing timestamps.")
    return dates


def _finite_numeric(series: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ContractViolation(f"{name} contains missing or non-numeric values.")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ContractViolation(f"{name} contains non-finite values.")
    return numeric


def _require_nonempty_strings(series: pd.Series, name: str) -> None:
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ContractViolation(f"{name} contains missing or empty identifiers.")


def _require_boolean_like(series: pd.Series, name: str) -> None:
    non_missing = series.dropna()
    if not non_missing.map(lambda value: isinstance(value, (bool, np.bool_)) or value in (0, 1)).all():
        raise ContractViolation(f"{name} must contain only boolean values.")


def _is_sorted(frame: pd.DataFrame, columns: list[str]) -> bool:
    if frame.empty:
        return True
    ordered_index = frame.sort_values(columns, kind="mergesort").index
    return bool(ordered_index.equals(frame.index))


def _date_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ContractViolation(f"{name} is not a valid timestamp.")
    return pd.Timestamp(parsed)


__all__ = [
    "ContractViolation",
    "FeatureFrameContract",
    "FoldSpec",
    "FrameValidationReport",
    "PROSPECTIVE_SEMANTICS",
    "ProbabilitySemantics",
    "ProbabilityUse",
    "ProspectiveProbabilityError",
    "RAW_DAILY_BAR_OPTIONAL_COLUMNS",
    "RAW_DAILY_BAR_REQUIRED_COLUMNS",
    "RawDailyBarContract",
    "StageRunResult",
    "StateProbabilityContract",
]
