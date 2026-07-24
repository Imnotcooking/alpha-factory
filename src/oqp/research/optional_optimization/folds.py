"""Expanding chronological inner folds with purge and embargo gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oqp.research.optional_optimization.contracts import Phase8ExperimentSpec


@dataclass(frozen=True, slots=True)
class Phase8Fold:
    fold_id: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    purge_dates: tuple[str, ...]
    embargo_dates: tuple[str, ...]
    training_periods: int
    validation_periods: int
    training_rows: int
    validation_rows: int
    training_data: pd.DataFrame
    validation_data: pd.DataFrame

    def to_record(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "purge_dates": list(self.purge_dates),
            "embargo_dates": list(self.embargo_dates),
            "training_periods": self.training_periods,
            "validation_periods": self.validation_periods,
            "training_rows": self.training_rows,
            "validation_rows": self.validation_rows,
        }


def build_phase8_folds(
    data: pd.DataFrame,
    spec: Phase8ExperimentSpec,
) -> tuple[Phase8Fold, ...]:
    config = spec.fold_config
    if config.date_col not in data.columns:
        raise ValueError(f"optimization data is missing {config.date_col!r}")
    work = data.copy()
    work[config.date_col] = pd.to_datetime(work[config.date_col], errors="raise")
    holdout_start = pd.Timestamp(spec.holdout_start).normalize()
    development = work.loc[work[config.date_col].dt.normalize().lt(holdout_start)].copy()
    dates = pd.Index(
        development[config.date_col].dt.normalize().drop_duplicates().sort_values()
    )
    first_validation = len(dates) - config.fold_count * config.validation_periods
    required_gap = config.purge_periods + config.embargo_periods
    if first_validation - required_gap < config.minimum_training_periods:
        required = (
            config.minimum_training_periods
            + required_gap
            + config.fold_count * config.validation_periods
        )
        raise ValueError(
            "insufficient development history for frozen Phase 8 folds: "
            f"need at least {required} distinct periods, found {len(dates)}"
        )

    folds: list[Phase8Fold] = []
    normalized_dates = development[config.date_col].dt.normalize()
    for index in range(config.fold_count):
        validation_start_position = (
            first_validation + index * config.validation_periods
        )
        validation_end_position = (
            validation_start_position + config.validation_periods
        )
        training_end_position = validation_start_position - required_gap
        purge_start = training_end_position
        embargo_start = purge_start + config.purge_periods
        training_dates = dates[:training_end_position]
        purge_dates = dates[purge_start:embargo_start]
        embargo_dates = dates[embargo_start:validation_start_position]
        validation_dates = dates[
            validation_start_position:validation_end_position
        ]
        training = development.loc[normalized_dates.isin(training_dates)].copy()
        validation = development.loc[
            normalized_dates.isin(validation_dates)
        ].copy()
        training.attrs.update(data.attrs)
        validation.attrs.update(data.attrs)
        folds.append(
            Phase8Fold(
                fold_id=f"inner_{index + 1:02d}",
                training_start=_date_text(training_dates[0]),
                training_end=_date_text(training_dates[-1]),
                validation_start=_date_text(validation_dates[0]),
                validation_end=_date_text(validation_dates[-1]),
                purge_dates=tuple(_date_text(value) for value in purge_dates),
                embargo_dates=tuple(_date_text(value) for value in embargo_dates),
                training_periods=len(training_dates),
                validation_periods=len(validation_dates),
                training_rows=len(training),
                validation_rows=len(validation),
                training_data=training,
                validation_data=validation,
            )
        )
    _validate_fold_geometry(tuple(folds), spec)
    return tuple(folds)


def _validate_fold_geometry(
    folds: tuple[Phase8Fold, ...], spec: Phase8ExperimentSpec
) -> None:
    holdout_start = pd.Timestamp(spec.holdout_start).normalize()
    for fold in folds:
        training_end = pd.Timestamp(fold.training_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        gap = tuple(pd.Timestamp(value) for value in fold.purge_dates) + tuple(
            pd.Timestamp(value) for value in fold.embargo_dates
        )
        if training_end >= validation_start:
            raise ValueError("Phase 8 training must precede validation")
        if validation_end >= holdout_start:
            raise ValueError("Phase 8 inner folds cannot enter the final holdout")
        if any(not training_end < value < validation_start for value in gap):
            raise ValueError("purge and embargo dates must lie inside the fold gap")


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


__all__ = ["Phase8Fold", "build_phase8_folds"]
