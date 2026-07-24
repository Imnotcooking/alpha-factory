"""Explicit signal-decision and position-holding policies for research runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


TEMPORAL_POLICY_VERSION = "1.0"

SIGNAL_EVERY_BAR = "every_bar"
SIGNAL_SESSION_CLOSE = "session_close"
SIGNAL_FIXED_INTERVAL = "fixed_interval"
SIGNAL_EVENT_DRIVEN = "event_driven"
VALID_SIGNAL_FREQUENCIES = {
    SIGNAL_EVERY_BAR,
    SIGNAL_SESSION_CLOSE,
    SIGNAL_FIXED_INTERVAL,
    SIGNAL_EVENT_DRIVEN,
}

HOLD_UNTIL_NEXT_DECISION = "until_next_decision"
HOLD_FIXED_PERIOD = "fixed_period"
HOLD_FACTOR_MANAGED = "factor_managed"
HOLD_SESSION_FLAT = "session_flat"
VALID_HOLDING_MODES = {
    HOLD_UNTIL_NEXT_DECISION,
    HOLD_FIXED_PERIOD,
    HOLD_FACTOR_MANAGED,
    HOLD_SESSION_FLAT,
}

VALID_DECISION_UNITS = {"bars", "minutes", "sessions", "events"}
VALID_HOLDING_UNITS = {"bars", "minutes", "sessions"}
VALID_ZERO_SIGNAL_ACTIONS = {"exit", "hold"}


@dataclass(frozen=True, slots=True)
class SignalHoldingPolicy:
    """Frozen rules for accepting signals and maintaining target positions."""

    policy_id: str
    data_frequency: str
    signal_frequency: str
    decision_interval: int
    decision_unit: str
    holding_mode: str
    holding_period: int | None = None
    holding_unit: str = "sessions"
    zero_signal_action: str = "exit"
    event_column: str | None = None
    source: str = "explicit"
    version: str = TEMPORAL_POLICY_VERSION

    def __post_init__(self) -> None:
        data_frequency = _normalize_data_frequency(self.data_frequency)
        signal_frequency = _normalize_token(self.signal_frequency)
        decision_unit = _normalize_token(self.decision_unit)
        holding_mode = _normalize_token(self.holding_mode)
        holding_unit = _normalize_token(self.holding_unit)
        zero_signal_action = _normalize_token(self.zero_signal_action)
        if signal_frequency not in VALID_SIGNAL_FREQUENCIES:
            raise ValueError(
                f"signal_frequency must be one of {sorted(VALID_SIGNAL_FREQUENCIES)}"
            )
        if self.decision_interval < 1:
            raise ValueError("decision_interval must be at least 1")
        if decision_unit not in VALID_DECISION_UNITS:
            raise ValueError(
                f"decision_unit must be one of {sorted(VALID_DECISION_UNITS)}"
            )
        if holding_mode not in VALID_HOLDING_MODES:
            raise ValueError(
                f"holding_mode must be one of {sorted(VALID_HOLDING_MODES)}"
            )
        if holding_unit not in VALID_HOLDING_UNITS:
            raise ValueError(
                f"holding_unit must be one of {sorted(VALID_HOLDING_UNITS)}"
            )
        if zero_signal_action not in VALID_ZERO_SIGNAL_ACTIONS:
            raise ValueError(
                "zero_signal_action must be 'exit' or 'hold'"
            )
        if holding_mode == HOLD_FIXED_PERIOD:
            if self.holding_period is None or int(self.holding_period) < 1:
                raise ValueError(
                    "fixed_period holding requires holding_period >= 1"
                )
        elif self.holding_period is not None:
            raise ValueError(
                "holding_period is only valid when holding_mode='fixed_period'"
            )
        if signal_frequency == SIGNAL_EVENT_DRIVEN and decision_unit != "events":
            raise ValueError("event_driven signals require decision_unit='events'")
        object.__setattr__(self, "data_frequency", data_frequency)
        object.__setattr__(self, "signal_frequency", signal_frequency)
        object.__setattr__(self, "decision_unit", decision_unit)
        object.__setattr__(self, "holding_mode", holding_mode)
        object.__setattr__(self, "holding_unit", holding_unit)
        object.__setattr__(self, "zero_signal_action", zero_signal_action)
        object.__setattr__(
            self,
            "holding_period",
            None if self.holding_period is None else int(self.holding_period),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalPolicySummary:
    total_rows: int
    decision_rows: int
    decision_rate: float
    target_change_count: int
    entry_count: int
    exit_count: int
    reversal_count: int
    active_rows: int
    realized_active_run_count: int
    realized_active_run_mean_bars: float
    realized_active_run_median_bars: float
    realized_active_run_max_bars: int
    policy_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_signal_holding_policy(
    frame: pd.DataFrame,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> SignalHoldingPolicy:
    """Resolve an explicit policy from strategy overrides and factor metadata."""

    metadata = frame.attrs.get("factor_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    contract = frame.attrs.get("factor_contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    explicit = dict(overrides or frame.attrs.get("temporal_policy_overrides") or {})

    data_frequency = _normalize_data_frequency(
        explicit.pop("data_frequency", None)
        or frame.attrs.get("data_frequency")
        or metadata.get("data_frequency")
        or _infer_frame_frequency(frame)
    )
    raw_signal_frequency = (
        explicit.pop("signal_frequency", None)
        or frame.attrs.get("signal_frequency")
        or metadata.get("signal_frequency")
    )
    signal_frequency, decision_interval, decision_unit = _parse_signal_frequency(
        raw_signal_frequency,
        data_frequency=data_frequency,
    )
    decision_interval = int(
        explicit.pop("decision_interval", decision_interval)
    )
    decision_unit = str(explicit.pop("decision_unit", decision_unit))

    raw_holding_mode = (
        explicit.pop("holding_mode", None)
        or frame.attrs.get("holding_mode")
        or metadata.get("holding_mode")
    )
    raw_holding_period = explicit.pop("holding_period", None)
    if raw_holding_period is None:
        raw_holding_period = frame.attrs.get("holding_period")
    if raw_holding_period is None:
        raw_holding_period = metadata.get("holding_period")
    holding_mode, holding_period = _parse_holding_rule(
        raw_holding_mode,
        raw_holding_period,
        execution_mode=str(
            contract.get("execution_mode")
            or frame.attrs.get("execution_mode")
            or ""
        ),
        frame=frame,
    )
    holding_unit = str(
        explicit.pop("holding_unit", None)
        or frame.attrs.get("holding_unit")
        or metadata.get("holding_unit")
        or ("bars" if data_frequency in {"intraday", "tick"} else "sessions")
    )
    zero_signal_action = str(
        explicit.pop("zero_signal_action", None)
        or frame.attrs.get("zero_signal_action")
        or metadata.get("zero_signal_action")
        or "exit"
    )
    event_column = explicit.pop("event_column", None)
    policy_id = str(
        explicit.pop("policy_id", None)
        or f"tmp_001_{data_frequency}_{signal_frequency}_{holding_mode}"
    )
    if explicit:
        raise ValueError(
            "Unknown temporal policy override(s): " + ", ".join(sorted(explicit))
        )
    source = (
        "explicit"
        if overrides or frame.attrs.get("temporal_policy_overrides")
        else "metadata_or_default"
    )
    return SignalHoldingPolicy(
        policy_id=policy_id,
        data_frequency=data_frequency,
        signal_frequency=signal_frequency,
        decision_interval=decision_interval,
        decision_unit=decision_unit,
        holding_mode=holding_mode,
        holding_period=holding_period,
        holding_unit=holding_unit,
        zero_signal_action=zero_signal_action,
        event_column=None if event_column in (None, "") else str(event_column),
        source=source,
    )


def ensure_signal_holding_policy(
    frame: pd.DataFrame,
    *,
    candidate_col: str,
    overrides: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if (
        not overrides
        and "signal_decision_row" in frame.columns
        and frame.attrs.get("temporal_policy_fingerprint")
        and frame.attrs.get("temporal_candidate_col") == candidate_col
    ):
        return frame
    policy = resolve_signal_holding_policy(frame, overrides=overrides)
    if (
        "signal_decision_row" in frame.columns
        and frame.attrs.get("temporal_policy_fingerprint") == policy.fingerprint
        and frame.attrs.get("temporal_candidate_col") == candidate_col
    ):
        return frame
    return apply_signal_holding_policy(
        frame,
        policy,
        candidate_col=candidate_col,
    )


def apply_signal_holding_policy(
    frame: pd.DataFrame,
    policy: SignalHoldingPolicy,
    *,
    candidate_col: str,
) -> pd.DataFrame:
    """Apply a causal decision schedule and holding rule to one target column."""

    if candidate_col not in frame.columns:
        raise ValueError(f"temporal policy candidate column is missing: {candidate_col}")
    out = _prepare_frame(frame)
    attrs = dict(frame.attrs)
    backup_col = f"pre_temporal_{candidate_col}"
    if backup_col not in out.columns:
        out[backup_col] = out[candidate_col]
    candidate = pd.to_numeric(out[backup_col], errors="coerce").fillna(0.0)
    decision_mask = _decision_mask(out, policy, candidate)
    if "liquidity_eligible" in out.columns:
        raw_liquidity = out["liquidity_eligible"]
        if pd.api.types.is_bool_dtype(raw_liquidity.dtype):
            liquidity_eligible = raw_liquidity.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(raw_liquidity.dtype):
            liquidity_eligible = (
                pd.to_numeric(raw_liquidity, errors="coerce")
                .fillna(0.0)
                .ne(0.0)
            )
        else:
            liquidity_eligible = (
                raw_liquidity.astype("string")
                .str.strip()
                .str.lower()
                .isin({"1", "true", "t", "yes", "y"})
            )
        liquidity_blocked = (~liquidity_eligible).astype(bool)
    else:
        liquidity_blocked = pd.Series(False, index=out.index, dtype=bool)
    decision_mask &= ~liquidity_blocked
    out["signal_decision_row"] = decision_mask
    out["signal_decision_reason"] = np.where(
        liquidity_blocked,
        "liquidity_blocked",
        np.where(decision_mask, "scheduled_decision", "between_decisions"),
    )

    if policy.holding_mode == HOLD_FACTOR_MANAGED:
        effective = candidate.copy()
        holding_age = _active_run_age(out, effective)
    else:
        accepted = candidate.where(decision_mask)
        if policy.zero_signal_action == "hold":
            accepted = accepted.mask(accepted.eq(0.0))
        segment = liquidity_blocked.groupby(out["ticker"], sort=False).cumsum()
        grouping = [out["ticker"], segment]
        if policy.holding_mode == HOLD_SESSION_FLAT:
            grouping.append(out["_temporal_session"])
        effective = accepted.groupby(grouping, sort=False).ffill().fillna(0.0)
        holding_age = _holding_age_since_decision(out, decision_mask, grouping, policy)
        if policy.holding_mode == HOLD_FIXED_PERIOD:
            effective = effective.where(
                holding_age.lt(int(policy.holding_period or 0)),
                0.0,
            )
        effective = effective.mask(liquidity_blocked, 0.0)

    out[candidate_col] = effective.astype(float)
    out["signal_holding_age"] = holding_age.fillna(0.0).astype(float)
    out["signal_target_changed_by_temporal_policy"] = ~np.isclose(
        candidate.to_numpy(dtype=float),
        effective.to_numpy(dtype=float),
        equal_nan=True,
    )
    out = out.sort_values("_temporal_row_order").drop(
        columns=["_temporal_row_order", "_temporal_session"]
    )
    out.index = frame.index
    out.attrs.update(attrs)
    return _attach_temporal_attrs(out, policy, candidate_col)


def temporal_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only rows on which the frozen policy accepted a new decision."""

    if "signal_decision_row" not in frame.columns:
        raise ValueError("frame has not been assessed by a temporal policy")
    mask = frame["signal_decision_row"].fillna(False)
    if "liquidity_eligible" in frame.columns:
        mask &= frame["liquidity_eligible"].fillna(False)
    out = frame.loc[mask].copy()
    out.attrs.update(frame.attrs)
    return out


def synchronize_temporal_targets(
    frame: pd.DataFrame,
    *,
    effective_col: str,
    target_columns: tuple[str, ...] = (
        "routed_target_weight",
        "final_target_weight",
        "target_weight",
        "signal",
    ),
) -> pd.DataFrame:
    """Copy one temporally enforced target into equivalent execution columns."""

    if effective_col not in frame.columns:
        raise ValueError(f"effective temporal target is missing: {effective_col}")
    out = frame.copy()
    out.attrs.update(frame.attrs)
    for column in target_columns:
        if column not in out.columns or column == effective_col:
            continue
        backup = f"pre_temporal_{column}"
        if backup not in out.columns:
            out[backup] = out[column]
        out[column] = out[effective_col]
    out.attrs["temporal_synchronized_target_columns"] = [
        column for column in target_columns if column in out.columns
    ]
    return out


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "ticker" not in out.columns and "symbol" in out.columns:
        out["ticker"] = out["symbol"].astype(str)
    if "ticker" not in out.columns:
        raise ValueError("temporal policy requires ticker or symbol")
    time_col = next(
        (
            column
            for column in ("datetime", "date", "timestamp")
            if column in out.columns
        ),
        None,
    )
    if time_col is None:
        raise ValueError("temporal policy requires date, datetime, or timestamp")
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if out[time_col].isna().any():
        raise ValueError("temporal policy found invalid timestamps")
    out["_temporal_row_order"] = np.arange(len(out))
    session_source = next(
        (
            column
            for column in ("trading_day", "economic_day", "session_date")
            if column in out.columns
        ),
        time_col,
    )
    out["_temporal_session"] = pd.to_datetime(
        out[session_source],
        errors="coerce",
    ).dt.normalize()
    return out.sort_values(["ticker", time_col, "_temporal_row_order"])


def _decision_mask(
    frame: pd.DataFrame,
    policy: SignalHoldingPolicy,
    candidate: pd.Series,
) -> pd.Series:
    if policy.signal_frequency == SIGNAL_EVERY_BAR:
        return pd.Series(True, index=frame.index)
    if policy.signal_frequency == SIGNAL_SESSION_CLOSE:
        return _last_row_mask(frame, ["ticker", "_temporal_session"])
    if policy.signal_frequency == SIGNAL_EVENT_DRIVEN:
        if policy.event_column:
            if policy.event_column not in frame.columns:
                raise ValueError(
                    f"temporal event column is missing: {policy.event_column}"
                )
            return frame[policy.event_column].fillna(False).astype(bool)
        previous = candidate.groupby(frame["ticker"], sort=False).shift(1)
        return candidate.ne(previous).fillna(True)
    if policy.signal_frequency != SIGNAL_FIXED_INTERVAL:
        raise ValueError(f"Unsupported signal frequency: {policy.signal_frequency}")
    interval = int(policy.decision_interval)
    if policy.decision_unit == "bars":
        ordinal = frame.groupby("ticker", sort=False).cumcount()
        return ordinal.mod(interval).eq(0)
    if policy.decision_unit == "sessions":
        session_rows = frame[["ticker", "_temporal_session"]].drop_duplicates()
        session_rows["_decision"] = (
            session_rows.groupby("ticker", sort=False)
            .cumcount()
            .mod(interval)
            .eq(0)
        )
        mapped = frame.merge(
            session_rows,
            on=["ticker", "_temporal_session"],
            how="left",
            validate="many_to_one",
        )["_decision"]
        mapped.index = frame.index
        return mapped & _last_row_mask(frame, ["ticker", "_temporal_session"])
    if policy.decision_unit == "minutes":
        time_col = next(
            column for column in ("datetime", "date", "timestamp") if column in frame.columns
        )
        bucket = frame[time_col].dt.floor(f"{interval}min")
        keyed = frame.assign(_temporal_bucket=bucket)
        return _last_row_mask(keyed, ["ticker", "_temporal_bucket"])
    raise ValueError(
        f"fixed_interval does not support decision_unit={policy.decision_unit!r}"
    )


def _holding_age_since_decision(
    frame: pd.DataFrame,
    decision_mask: pd.Series,
    grouping: list[pd.Series],
    policy: SignalHoldingPolicy,
) -> pd.Series:
    if policy.holding_unit == "bars":
        accepted_id = decision_mask.groupby(grouping, sort=False).cumsum()
        age = frame.groupby([*grouping, accepted_id], sort=False).cumcount()
        return age.where(accepted_id.gt(0), np.nan).astype(float)
    if policy.holding_unit == "sessions":
        session_ordinal = frame.groupby("ticker", sort=False)[
            "_temporal_session"
        ].transform(lambda values: pd.factorize(values, sort=False)[0])
        last_decision = session_ordinal.where(decision_mask).groupby(
            grouping,
            sort=False,
        ).ffill()
        return (session_ordinal - last_decision).where(last_decision.notna()).astype(float)
    if policy.holding_unit == "minutes":
        time_col = next(
            column for column in ("datetime", "date", "timestamp") if column in frame.columns
        )
        last_decision = frame[time_col].where(decision_mask).groupby(
            grouping,
            sort=False,
        ).ffill()
        return (
            (frame[time_col] - last_decision).dt.total_seconds() / 60.0
        ).where(last_decision.notna())
    raise ValueError(f"Unsupported holding unit: {policy.holding_unit}")


def _active_run_age(frame: pd.DataFrame, target: pd.Series) -> pd.Series:
    sign = np.sign(pd.to_numeric(target, errors="coerce").fillna(0.0))
    previous = sign.groupby(frame["ticker"], sort=False).shift(1)
    new_run = sign.ne(previous) | sign.eq(0.0)
    run_id = new_run.groupby(frame["ticker"], sort=False).cumsum()
    age = frame.groupby([frame["ticker"], run_id], sort=False).cumcount()
    return age.where(sign.ne(0.0), 0.0).astype(float)


def _attach_temporal_attrs(
    frame: pd.DataFrame,
    policy: SignalHoldingPolicy,
    candidate_col: str,
) -> pd.DataFrame:
    summary = _summarize_temporal_behavior(frame, candidate_col, policy)
    frame.attrs["temporal_policy"] = policy.to_dict()
    frame.attrs["temporal_policy_id"] = policy.policy_id
    frame.attrs["temporal_policy_fingerprint"] = policy.fingerprint
    frame.attrs["temporal_policy_summary"] = summary.to_dict()
    frame.attrs["temporal_candidate_col"] = candidate_col
    frame.attrs["temporal_policy_status"] = "enforced"
    return frame


def _summarize_temporal_behavior(
    frame: pd.DataFrame,
    candidate_col: str,
    policy: SignalHoldingPolicy,
) -> TemporalPolicySummary:
    ordered = frame.copy()
    time_col = next(
        column for column in ("datetime", "date", "timestamp") if column in ordered.columns
    )
    ordered = ordered.sort_values(["ticker", time_col])
    target = pd.to_numeric(ordered[candidate_col], errors="coerce").fillna(0.0)
    previous = target.groupby(ordered["ticker"], sort=False).shift(1).fillna(0.0)
    current_sign = np.sign(target)
    previous_sign = np.sign(previous)
    entries = current_sign.ne(0.0) & previous_sign.eq(0.0)
    exits = current_sign.eq(0.0) & previous_sign.ne(0.0)
    reversals = current_sign.ne(0.0) & previous_sign.ne(0.0) & current_sign.ne(previous_sign)
    changed = target.sub(previous).abs().gt(1e-12)

    active = current_sign.ne(0.0)
    block_start = (
        active.ne(active.groupby(ordered["ticker"], sort=False).shift(1).fillna(False))
        | current_sign.ne(previous_sign)
    )
    block_id = block_start.groupby(ordered["ticker"], sort=False).cumsum()
    active_runs = (
        pd.DataFrame(
            {
                "ticker": ordered["ticker"].astype(str),
                "block": block_id,
                "active": active,
            }
        )
        .loc[active]
        .groupby(["ticker", "block"], sort=False)
        .size()
    )
    run_count = int(len(active_runs))
    return TemporalPolicySummary(
        total_rows=int(len(frame)),
        decision_rows=int(frame["signal_decision_row"].fillna(False).sum()),
        decision_rate=(
            float(frame["signal_decision_row"].fillna(False).mean())
            if len(frame)
            else 0.0
        ),
        target_change_count=int(changed.sum()),
        entry_count=int(entries.sum()),
        exit_count=int(exits.sum()),
        reversal_count=int(reversals.sum()),
        active_rows=int(active.sum()),
        realized_active_run_count=run_count,
        realized_active_run_mean_bars=(float(active_runs.mean()) if run_count else 0.0),
        realized_active_run_median_bars=(float(active_runs.median()) if run_count else 0.0),
        realized_active_run_max_bars=(int(active_runs.max()) if run_count else 0),
        policy_fingerprint=policy.fingerprint,
    )


def _parse_signal_frequency(
    value: Any,
    *,
    data_frequency: str,
) -> tuple[str, int, str]:
    raw = _normalize_token(value)
    if not raw:
        return (
            (SIGNAL_SESSION_CLOSE, 1, "sessions")
            if data_frequency == "daily"
            else (SIGNAL_EVERY_BAR, 1, "bars")
        )
    if raw in {"daily", "daily_close", "session_close", "every_session_close"}:
        return SIGNAL_SESSION_CLOSE, 1, "sessions"
    if raw in {"tick", "every_tick", "every_bar", "each_bar"}:
        return SIGNAL_EVERY_BAR, 1, "bars"
    minute_match = re.search(r"(?:completed_)?(\d+)\s*(?:min|minute)", raw)
    if minute_match:
        return SIGNAL_FIXED_INTERVAL, int(minute_match.group(1)), "minutes"
    session_match = re.search(r"every_(\d+)_sessions?", raw)
    if session_match:
        return SIGNAL_FIXED_INTERVAL, int(session_match.group(1)), "sessions"
    if "event" in raw:
        return SIGNAL_EVENT_DRIVEN, 1, "events"
    if raw in VALID_SIGNAL_FREQUENCIES:
        unit = "events" if raw == SIGNAL_EVENT_DRIVEN else "bars"
        return raw, 1, unit
    raise ValueError(f"Unrecognized signal_frequency={value!r}")


def _parse_holding_rule(
    raw_mode: Any,
    raw_period: Any,
    *,
    execution_mode: str,
    frame: pd.DataFrame,
) -> tuple[str, int | None]:
    mode = _normalize_token(raw_mode)
    if not mode and isinstance(raw_period, str):
        period_token = _normalize_token(raw_period)
        if period_token in VALID_HOLDING_MODES:
            mode = period_token
            raw_period = None
    if not mode:
        stateful = any("active_state" in str(column) for column in frame.columns)
        mode = (
            HOLD_FACTOR_MANAGED
            if _normalize_token(execution_mode) in {"direct", "statarb"} or stateful
            else HOLD_UNTIL_NEXT_DECISION
        )
    aliases = {
        "until_next_signal": HOLD_UNTIL_NEXT_DECISION,
        "until_signal_change": HOLD_FACTOR_MANAGED,
        "stateful": HOLD_FACTOR_MANAGED,
        "fixed": HOLD_FIXED_PERIOD,
        "intraday_flat": HOLD_SESSION_FLAT,
    }
    mode = aliases.get(mode, mode)
    period = None if raw_period in (None, "") else int(raw_period)
    return mode, period


def _infer_frame_frequency(frame: pd.DataFrame) -> str:
    time_col = next(
        (
            column
            for column in ("datetime", "date", "timestamp")
            if column in frame.columns
        ),
        None,
    )
    if time_col is None:
        return "unknown"
    values = pd.to_datetime(frame[time_col], errors="coerce").dropna()
    if values.empty:
        return "unknown"
    return "intraday" if values.dt.normalize().ne(values).any() else "daily"


def _normalize_data_frequency(value: Any) -> str:
    token = _normalize_token(value)
    aliases = {
        "1d": "daily",
        "day": "daily",
        "minute": "intraday",
        "minutely": "intraday",
    }
    return aliases.get(token, token or "unknown")


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _last_row_mask(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    return ~frame.duplicated(keys, keep="last")


__all__ = [
    "HOLD_FACTOR_MANAGED",
    "HOLD_FIXED_PERIOD",
    "HOLD_SESSION_FLAT",
    "HOLD_UNTIL_NEXT_DECISION",
    "SIGNAL_EVENT_DRIVEN",
    "SIGNAL_EVERY_BAR",
    "SIGNAL_FIXED_INTERVAL",
    "SIGNAL_SESSION_CLOSE",
    "SignalHoldingPolicy",
    "TEMPORAL_POLICY_VERSION",
    "TemporalPolicySummary",
    "apply_signal_holding_policy",
    "ensure_signal_holding_policy",
    "resolve_signal_holding_policy",
    "synchronize_temporal_targets",
    "temporal_metric_rows",
]
