from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


SplitMode = Literal["auto", "date", "ratio"]
PurgeUnit = Literal["auto", "days", "timestamps", "rows"]


@dataclass(frozen=True)
class EvaluationSplitPolicy:
    mode: SplitMode = "auto"
    split_date: str | pd.Timestamp = "2023-01-01"
    validation_fraction: float = 0.60
    min_valid_rows: int = 10
    min_day_split_count: int = 3
    purge_periods: int = 0
    embargo_periods: int = 0
    purge_unit: PurgeUnit = "auto"


@dataclass(frozen=True)
class EvaluationSplitResult:
    validation_data: pd.DataFrame
    holdout_data: pd.DataFrame
    crisis_data: pd.DataFrame
    split_mode: str
    split_boundary: str
    validation_rows: int
    holdout_rows: int
    crisis_rows: int
    purge_periods: int = 0
    embargo_periods: int = 0
    purge_unit: str = "none"
    purged_rows: int = 0
    embargoed_rows: int = 0


def build_chronological_split(
    df: pd.DataFrame,
    split_date: str | pd.Timestamp = "2023-01-01",
    crisis_period: tuple | None = None,
    signal_col: str = "factor_score",
    return_col: str = "forward_return",
    mode: SplitMode = "auto",
    validation_fraction: float = 0.60,
    purge_periods: int | None = 0,
    embargo_periods: int | None = 0,
    purge_unit: PurgeUnit = "auto",
) -> EvaluationSplitResult:
    policy = EvaluationSplitPolicy(
        mode=_normalize_mode(mode),
        split_date=split_date,
        validation_fraction=float(validation_fraction),
        purge_periods=_nonnegative_int(purge_periods),
        embargo_periods=_nonnegative_int(embargo_periods),
        purge_unit=_normalize_purge_unit(purge_unit),
    )
    if df.empty or "date" not in df.columns:
        empty = df.iloc[0:0]
        return EvaluationSplitResult(empty, empty, empty, "empty", "", 0, 0, 0)

    work = df.copy()
    work["_split_date_value"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["_split_date_value"])
    if work.empty:
        empty = df.iloc[0:0]
        return EvaluationSplitResult(empty, empty, empty, "empty", "", 0, 0, 0)

    crisis_mask = _build_crisis_mask(work["_split_date_value"], crisis_period)
    crisis_data = work.loc[crisis_mask].drop(columns=["_split_date_value"])
    base = work.loc[~crisis_mask].copy()

    if base.empty:
        empty = df.iloc[0:0]
        return EvaluationSplitResult(
            empty,
            empty,
            crisis_data,
            "crisis_only",
            "",
            0,
            0,
            len(crisis_data),
        )

    if policy.mode in {"date", "auto"}:
        date_result = _split_by_date(base, policy, crisis_data, signal_col, return_col)
        if policy.mode == "date" or _split_has_evidence(date_result, policy):
            return date_result

    return _split_by_ratio(base, policy, crisis_data, signal_col, return_col)


def _normalize_mode(mode: str | None) -> SplitMode:
    value = (mode or "auto").strip().lower()
    if value not in {"auto", "date", "ratio"}:
        raise ValueError(f"Unknown split mode: {mode!r}")
    return value  # type: ignore[return-value]


def _normalize_purge_unit(unit: str | None) -> PurgeUnit:
    value = (unit or "auto").strip().lower()
    aliases = {
        "time": "timestamps",
        "timestamp": "timestamps",
        "bar": "timestamps",
        "bars": "timestamps",
        "day": "days",
        "date": "days",
        "row": "rows",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "days", "timestamps", "rows"}:
        raise ValueError(f"Unknown purge unit: {unit!r}")
    return value  # type: ignore[return-value]


def _nonnegative_int(value: int | float | str | None) -> int:
    if value is None:
        return 0
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 0


def _build_crisis_mask(dates: pd.Series, crisis_period: tuple | None) -> pd.Series:
    mask = pd.Series(False, index=dates.index)
    if not crisis_period or len(crisis_period) != 2:
        return mask
    start = pd.to_datetime(crisis_period[0], errors="coerce")
    end = pd.to_datetime(crisis_period[1], errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return mask
    if end < start:
        start, end = end, start
    return dates.between(start, end, inclusive="both")


def _split_by_date(
    base: pd.DataFrame,
    policy: EvaluationSplitPolicy,
    crisis_data: pd.DataFrame,
    signal_col: str,
    return_col: str,
) -> EvaluationSplitResult:
    boundary = pd.to_datetime(policy.split_date)
    return _finalize_split(
        base,
        base["_split_date_value"] < boundary,
        base["_split_date_value"] >= boundary,
        crisis_data,
        "date",
        boundary.isoformat() if boundary.time() != pd.Timestamp(0).time() else boundary.strftime("%Y-%m-%d"),
        policy,
        signal_col,
        return_col,
    )


def _split_by_ratio(
    base: pd.DataFrame,
    policy: EvaluationSplitPolicy,
    crisis_data: pd.DataFrame,
    signal_col: str,
    return_col: str,
) -> EvaluationSplitResult:
    fraction = min(max(policy.validation_fraction, 0.05), 0.95)
    ordered = base.sort_values("_split_date_value", kind="mergesort")
    day_values = ordered["_split_date_value"].dt.normalize()
    unique_days = pd.Index(day_values.dropna().unique()).sort_values()

    if len(unique_days) >= policy.min_day_split_count:
        split_idx = int(np.floor(len(unique_days) * fraction))
        split_idx = min(max(split_idx, 1), len(unique_days) - 1)
        validation_days = set(unique_days[:split_idx])
        boundary_day = pd.Timestamp(unique_days[split_idx])
        validation_idx = ordered.index[day_values.isin(validation_days)]
        split_mode = "ratio_days"
        split_boundary = boundary_day.strftime("%Y-%m-%d")
    else:
        split_idx = int(np.floor(len(ordered) * fraction))
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        validation_idx = ordered.index[:split_idx]
        split_mode = "ratio_rows"
        split_boundary = pd.Timestamp(ordered["_split_date_value"].iloc[split_idx]).isoformat()

    validation_mask = pd.Series(False, index=base.index)
    validation_mask.loc[validation_idx] = True
    holdout_mask = ~validation_mask
    return _finalize_split(
        base,
        validation_mask,
        holdout_mask,
        crisis_data,
        split_mode,
        split_boundary,
        policy,
        signal_col,
        return_col,
    )


def _finalize_split(
    base: pd.DataFrame,
    validation_mask: pd.Series,
    holdout_mask: pd.Series,
    crisis_data: pd.DataFrame,
    split_mode: str,
    split_boundary: str,
    policy: EvaluationSplitPolicy,
    signal_col: str,
    return_col: str,
) -> EvaluationSplitResult:
    validation = base.loc[validation_mask].copy()
    holdout = base.loc[holdout_mask].copy()
    validation, holdout, purged_rows, embargoed_rows, effective_unit = _apply_purge_embargo(
        base,
        validation,
        holdout,
        policy,
    )
    validation = validation.drop(columns=["_split_date_value"])
    holdout = holdout.drop(columns=["_split_date_value"])
    return EvaluationSplitResult(
        validation,
        holdout,
        crisis_data,
        _split_mode_with_gap(split_mode, policy),
        split_boundary,
        _valid_row_count(validation, signal_col, return_col),
        _valid_row_count(holdout, signal_col, return_col),
        len(crisis_data),
        policy.purge_periods,
        policy.embargo_periods,
        effective_unit,
        purged_rows,
        embargoed_rows,
    )


def _split_mode_with_gap(split_mode: str, policy: EvaluationSplitPolicy) -> str:
    if policy.purge_periods <= 0 and policy.embargo_periods <= 0:
        return split_mode
    return f"{split_mode}_purged"


def _apply_purge_embargo(
    base: pd.DataFrame,
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    policy: EvaluationSplitPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, str]:
    effective_unit = _resolve_effective_purge_unit(base, policy)
    purged_idx = _period_indices(validation, policy.purge_periods, effective_unit, side="tail")
    embargoed_idx = _period_indices(holdout, policy.embargo_periods, effective_unit, side="head")
    if len(purged_idx):
        validation = validation.drop(index=purged_idx)
    if len(embargoed_idx):
        holdout = holdout.drop(index=embargoed_idx)
    return validation, holdout, len(purged_idx), len(embargoed_idx), effective_unit


def _resolve_effective_purge_unit(base: pd.DataFrame, policy: EvaluationSplitPolicy) -> str:
    if policy.purge_periods <= 0 and policy.embargo_periods <= 0:
        return "none"
    if policy.purge_unit != "auto":
        return policy.purge_unit
    dates = pd.to_datetime(base["_split_date_value"], errors="coerce")
    if bool(dates.dt.normalize().ne(dates).any()):
        return "timestamps"
    return "days"


def _period_indices(df: pd.DataFrame, periods: int, unit: str, *, side: Literal["head", "tail"]) -> pd.Index:
    if periods <= 0 or df.empty:
        return pd.Index([])
    ordered = df.sort_values("_split_date_value", kind="mergesort")
    if unit == "rows":
        return ordered.index[:periods] if side == "head" else ordered.index[-periods:]
    if unit == "days":
        keys = ordered["_split_date_value"].dt.normalize()
    else:
        keys = ordered["_split_date_value"]
    unique_keys = pd.Index(keys.dropna().unique()).sort_values()
    if unique_keys.empty:
        return pd.Index([])
    selected = unique_keys[:periods] if side == "head" else unique_keys[-periods:]
    return ordered.index[keys.isin(set(selected))]


def _split_has_evidence(result: EvaluationSplitResult, policy: EvaluationSplitPolicy) -> bool:
    return (
        result.validation_rows >= policy.min_valid_rows
        and result.holdout_rows >= policy.min_valid_rows
    )


def _valid_row_count(df: pd.DataFrame, signal_col: str, return_col: str) -> int:
    if df.empty or signal_col not in df.columns or return_col not in df.columns:
        return 0
    valid = df[[signal_col, return_col]].copy()
    valid[signal_col] = pd.to_numeric(valid[signal_col], errors="coerce")
    valid[return_col] = pd.to_numeric(valid[return_col], errors="coerce")
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    return int(len(valid))
