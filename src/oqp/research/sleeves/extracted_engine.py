"""Canonical execution adapters for extracted sleeve contracts.

Extracted sleeves preserve small lifecycle rules that were removed from legacy
hybrid factor implementations.  Only rule families registered in this module
may execute.  The adapters consume an already-causal factor panel and never
inspect realised or forward returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from oqp.research.sleeves.extracted import ExtractedSleeveConfig


class ExtractedSleeveAlignmentError(ValueError):
    """Raised when an extracted sleeve lacks a causal input attestation."""


@dataclass(frozen=True, slots=True)
class ExtractedSleeveExecutionResult:
    """Deterministic target positions and diagnostics for an extracted rule."""

    config: ExtractedSleeveConfig
    positions: pd.DataFrame
    daily_summary: pd.DataFrame


ExtractedSleeveAdapter = Callable[
    [pd.DataFrame, ExtractedSleeveConfig],
    ExtractedSleeveExecutionResult,
]


def supports_extracted_sleeve_execution(
    config: ExtractedSleeveConfig,
) -> bool:
    """Return whether a config opts into a registered canonical adapter."""

    return bool(config.execution_supported) and (
        config.rule_family in _EXTRACTED_SLEEVE_EXECUTION_ADAPTERS
    )


def build_extracted_sleeve_targets(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Dispatch an extracted sleeve through its registered rule-family adapter."""

    if not config.execution_supported:
        raise ValueError(
            f"{config.sleeve_id} declares execution_supported=False"
        )
    adapter = _EXTRACTED_SLEEVE_EXECUTION_ADAPTERS.get(config.rule_family)
    if adapter is None:
        raise ValueError(
            f"{config.sleeve_id} has no registered execution adapter for "
            f"rule_family={config.rule_family!r}"
        )
    return adapter(frame, config)


def _build_opposite_event_state(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Persist each signed event until a later nonzero event updates its state.

    A zero or missing score is deliberately a no-event row, not a flat signal.
    A positive/negative event updates the state to +1/-1 on that same decision
    row.  The current active states are equally weighted by absolute gross and
    then independently capped; clipped gross is retained as cash.
    """

    if not (
        bool(frame.attrs.get("causal_signal_alignment_verified"))
        or bool(frame.attrs.get("causal_return_alignment_verified"))
    ):
        raise ExtractedSleeveAlignmentError(
            "extracted sleeve construction requires an upstream causal signal "
            "attestation"
        )
    _validate_opposite_event_contract(config)

    required = {
        config.date_col,
        config.product_col,
        *config.required_signal_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"extracted sleeve input is missing columns: {missing}"
        )

    source_attrs = dict(frame.attrs)
    out = frame.copy()
    out[config.date_col] = pd.to_datetime(
        out[config.date_col], errors="coerce"
    ).dt.normalize()
    out[config.product_col] = (
        out[config.product_col].astype("string").str.strip()
    )
    invalid_keys = (
        out[config.date_col].isna()
        | out[config.product_col].isna()
        | out[config.product_col].eq("")
    )
    if invalid_keys.any():
        raise ValueError(
            "extracted sleeve input contains invalid date/product keys"
        )
    if out.duplicated([config.date_col, config.product_col]).any():
        raise ValueError(
            "extracted sleeve input requires unique date/product rows"
        )

    out[config.signal_col] = pd.to_numeric(
        out[config.signal_col], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(
        [config.product_col, config.date_col],
        kind="mergesort",
    ).reset_index(drop=True)

    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_event_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["signal_neutral"] = oriented_signal.eq(0.0)
    out["state_update_event"] = oriented_signal.notna() & oriented_signal.ne(
        0.0
    )
    out["event_direction"] = np.sign(oriented_signal).where(
        out["state_update_event"],
        0.0,
    )

    event_state = out["event_direction"].where(out["state_update_event"])
    out["directional_state"] = (
        event_state.groupby(out[config.product_col], sort=False)
        .ffill()
        .fillna(0.0)
        .astype(float)
    )
    out["prior_directional_state"] = (
        out.groupby(config.product_col, sort=False)["directional_state"]
        .shift(1)
        .fillna(0.0)
    )
    out["state_transition_event"] = out["directional_state"].ne(
        out["prior_directional_state"]
    )
    out["state_entry_event"] = (
        out["prior_directional_state"].eq(0.0)
        & out["directional_state"].ne(0.0)
    )
    out["state_flip_event"] = (
        out["prior_directional_state"]
        * out["directional_state"]
    ).lt(0.0)
    out["state_preserved_without_event"] = (
        ~out["state_update_event"]
        & out["directional_state"].ne(0.0)
    )
    state_segment = out["state_transition_event"].groupby(
        out[config.product_col],
        sort=False,
    ).cumsum()
    out["state_age_periods"] = (
        out.groupby(
            [out[config.product_col], state_segment],
            sort=False,
        )
        .cumcount()
        .add(1)
        .where(out["directional_state"].ne(0.0), 0)
        .astype(int)
    )

    parameters = dict(config.parameters or {})
    target_gross = float(parameters["target_gross_exposure"])
    max_weight = parameters.get("max_weight_per_contract")
    max_weight = None if max_weight is None else float(max_weight)

    active = out["directional_state"].ne(0.0)
    active_count = active.groupby(out[config.date_col], sort=False).transform(
        "sum"
    )
    out["active_state_count"] = active_count.astype(int)
    denominator = active_count.where(active_count.gt(0), np.nan)
    out["uncapped_target_weight"] = (
        out["directional_state"] / denominator * target_gross
    ).fillna(0.0)
    if max_weight is None:
        out["contract_cap_bound"] = False
        out[config.output_col] = out["uncapped_target_weight"]
    else:
        out["contract_cap_bound"] = out[
            "uncapped_target_weight"
        ].abs().gt(max_weight + 1e-15)
        out[config.output_col] = out["uncapped_target_weight"].clip(
            -max_weight,
            max_weight,
        )

    out["decision_target_weight"] = out[config.output_col]
    out["held_target_weight"] = out[config.output_col]
    out["selection_side"] = np.select(
        [
            out["directional_state"].gt(0.0),
            out["directional_state"].lt(0.0),
        ],
        ["long", "short"],
        default="flat",
    )
    out["prior_target_weight"] = (
        out.groupby(config.product_col, sort=False)[config.output_col]
        .shift(1)
        .fillna(0.0)
    )
    out["target_turnover"] = (
        out[config.output_col] - out["prior_target_weight"]
    ).abs()

    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)
    daily = _opposite_event_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_opposite_event_state_v1",
        "zero_signal_action": "preserve_state",
        "missing_signal_action": "preserve_state",
        "nonzero_signal_action": "update_state_from_oriented_sign",
        "state_update_timing": "same_decision_row",
        "normalization": "equal_weight_active_signs",
        "rescale_after_contract_cap": False,
        "future_return_used": False,
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _validate_opposite_event_contract(
    config: ExtractedSleeveConfig,
) -> None:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "construction_geometry": "time_series_stateful",
        "expression": "directional",
        "construction": "opposite_event_state",
        "normalization": "equal_weight_active_signs",
    }
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} opposite-event adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )
    for field in ("zero_signal_action", "missing_signal_action"):
        action = str(parameters.get(field) or "preserve_state").strip().lower()
        if action != "preserve_state":
            raise ValueError(
                f"{config.sleeve_id} opposite-event adapter requires "
                f"{field}='preserve_state'"
            )
    if bool(parameters.get("rescale_after_contract_cap", False)):
        raise ValueError(
            f"{config.sleeve_id} opposite-event adapter cannot rescale after "
            "the contract cap"
        )

    try:
        target_gross = float(parameters["target_gross_exposure"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires finite target_gross_exposure"
        ) from exc
    if not np.isfinite(target_gross) or not 0.0 < target_gross <= 2.0:
        raise ValueError(
            f"{config.sleeve_id} target_gross_exposure must be in (0, 2]"
        )

    max_weight = parameters.get("max_weight_per_contract")
    if max_weight is not None:
        try:
            max_weight = float(max_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be numeric"
            ) from exc
        if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be in (0, 1]"
            )


def _opposite_event_daily_summary(
    positions: pd.DataFrame,
    *,
    config: ExtractedSleeveConfig,
    target_gross: float,
) -> pd.DataFrame:
    output_col = config.output_col
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=(output_col, lambda values: float(values.abs().sum())),
        net_exposure=(output_col, "sum"),
        long_exposure=(
            output_col,
            lambda values: float(values.clip(lower=0.0).sum()),
        ),
        short_exposure=(
            output_col,
            lambda values: float((-values.clip(upper=0.0)).sum()),
        ),
        target_turnover=("target_turnover", "sum"),
        active_products=(
            "directional_state",
            lambda values: int(values.ne(0.0).sum()),
        ),
        long_products=(
            "directional_state",
            lambda values: int(values.gt(0.0).sum()),
        ),
        short_products=(
            "directional_state",
            lambda values: int(values.lt(0.0).sum()),
        ),
        state_updates=("state_update_event", "sum"),
        state_entries=("state_entry_event", "sum"),
        state_flips=("state_flip_event", "sum"),
        states_preserved=("state_preserved_without_event", "sum"),
        missing_signals=("signal_missing", "sum"),
        neutral_signals=("signal_neutral", "sum"),
        contract_cap_count=("contract_cap_bound", "sum"),
    ).reset_index()
    daily["gross_realization"] = (
        daily["gross_exposure"] / target_gross
    )
    return daily


def _build_residual_event_ttl(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Run a residual-dislocation entry/decay/TTL state machine.

    The input score is already aligned to its factor-owned return row.  This
    adapter therefore updates the state on that same decision row and never
    shifts either the score or the target.  Entry age starts at one.  An active
    lifecycle may occupy at most ``holding_periods`` observed product sessions;
    the old state is flat before a would-be age ``holding_periods + 1`` target.
    """

    if not (
        bool(frame.attrs.get("causal_signal_alignment_verified"))
        or bool(frame.attrs.get("causal_return_alignment_verified"))
    ):
        raise ExtractedSleeveAlignmentError(
            "extracted sleeve construction requires an upstream causal signal "
            "attestation"
        )
    _validate_residual_event_ttl_contract(config)

    required = {
        config.date_col,
        config.product_col,
        *config.required_signal_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"extracted sleeve input is missing columns: {missing}"
        )

    source_attrs = dict(frame.attrs)
    out = frame.copy()
    out[config.date_col] = pd.to_datetime(
        out[config.date_col],
        errors="coerce",
    ).dt.normalize()
    out[config.product_col] = (
        out[config.product_col].astype("string").str.strip()
    )
    invalid_keys = (
        out[config.date_col].isna()
        | out[config.product_col].isna()
        | out[config.product_col].eq("")
    )
    if invalid_keys.any():
        raise ValueError(
            "extracted sleeve input contains invalid date/product keys"
        )
    if out.duplicated([config.date_col, config.product_col]).any():
        raise ValueError(
            "extracted sleeve input requires unique date/product rows"
        )

    out[config.signal_col] = pd.to_numeric(
        out[config.signal_col],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(
        [config.product_col, config.date_col],
        kind="mergesort",
    ).reset_index(drop=True)

    parameters = dict(config.parameters or {})
    entry_abs_z = float(parameters["entry_abs_z"])
    exit_abs_z = float(parameters["exit_abs_z"])
    holding_periods = int(parameters["holding_periods"])
    target_gross = float(parameters["target_gross_exposure"])
    max_weight = parameters.get("max_weight_per_contract")
    max_weight = None if max_weight is None else float(max_weight)

    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_event_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["entry_threshold_event"] = (
        oriented_signal.notna() & oriented_signal.abs().ge(entry_abs_z)
    )
    out["decay_exit_threshold_event"] = (
        oriented_signal.notna() & oriented_signal.abs().le(exit_abs_z)
    )
    out["entry_event_direction"] = np.sign(oriented_signal).where(
        out["entry_threshold_event"],
        0.0,
    )

    row_count = len(out)
    directional_state = np.zeros(row_count, dtype=float)
    prior_directional_state = np.zeros(row_count, dtype=float)
    state_age_periods = np.zeros(row_count, dtype=int)
    prior_state_age_periods = np.zeros(row_count, dtype=int)
    state_entry_event = np.zeros(row_count, dtype=bool)
    state_flip_event = np.zeros(row_count, dtype=bool)
    decay_exit_event = np.zeros(row_count, dtype=bool)
    ttl_exit_event = np.zeros(row_count, dtype=bool)
    same_direction_entry_ignored = np.zeros(row_count, dtype=bool)
    state_preserved_on_missing = np.zeros(row_count, dtype=bool)
    lifecycle_exit_reason = np.full(row_count, "", dtype=object)

    for _, index in out.groupby(config.product_col, sort=False).groups.items():
        state = 0.0
        age = 0
        for position in index:
            prior_directional_state[position] = state
            prior_state_age_periods[position] = age
            signal = oriented_signal.iat[position]
            signal_is_missing = bool(pd.isna(signal))
            entry_event = bool(out["entry_threshold_event"].iat[position])
            decay_event = bool(
                out["decay_exit_threshold_event"].iat[position]
            )
            entry_direction = (
                float(np.sign(signal)) if entry_event else 0.0
            )

            if state != 0.0:
                same_direction = (
                    entry_event and entry_direction == state
                )
                opposite_direction = (
                    entry_event and entry_direction == -state
                )
                same_direction_entry_ignored[position] = same_direction

                if decay_event:
                    state = 0.0
                    age = 0
                    decay_exit_event[position] = True
                    lifecycle_exit_reason[position] = "decay_threshold"
                elif opposite_direction:
                    state = entry_direction
                    age = 1
                    state_flip_event[position] = True
                    lifecycle_exit_reason[position] = "opposite_entry_flip"
                elif age >= holding_periods:
                    state = 0.0
                    age = 0
                    ttl_exit_event[position] = True
                    lifecycle_exit_reason[position] = "ttl_expiry"
                else:
                    age += 1
                    if signal_is_missing:
                        state_preserved_on_missing[position] = True
            elif entry_event:
                state = entry_direction
                age = 1
                state_entry_event[position] = True

            directional_state[position] = state
            state_age_periods[position] = age

    out["prior_directional_state"] = prior_directional_state
    out["prior_state_age_periods"] = prior_state_age_periods
    out["directional_state"] = directional_state
    out["state_age_periods"] = state_age_periods
    out["state_entry_event"] = state_entry_event
    out["state_flip_event"] = state_flip_event
    out["decay_exit_event"] = decay_exit_event
    out["ttl_exit_event"] = ttl_exit_event
    out["same_direction_entry_ignored"] = (
        same_direction_entry_ignored
    )
    out["state_preserved_on_missing"] = state_preserved_on_missing
    out["lifecycle_exit_reason"] = lifecycle_exit_reason
    out["state_transition_event"] = out["directional_state"].ne(
        out["prior_directional_state"]
    )
    out["ttl_due_on_decision_row"] = (
        out["prior_directional_state"].ne(0.0)
        & out["prior_state_age_periods"].ge(holding_periods)
    )

    active = out["directional_state"].ne(0.0)
    active_count = active.groupby(out[config.date_col], sort=False).transform(
        "sum"
    )
    out["active_state_count"] = active_count.astype(int)
    denominator = active_count.where(active_count.gt(0), np.nan)
    out["uncapped_target_weight"] = (
        out["directional_state"] / denominator * target_gross
    ).fillna(0.0)
    if max_weight is None:
        out["contract_cap_bound"] = False
        out[config.output_col] = out["uncapped_target_weight"]
    else:
        out["contract_cap_bound"] = out[
            "uncapped_target_weight"
        ].abs().gt(max_weight + 1e-15)
        out[config.output_col] = out["uncapped_target_weight"].clip(
            -max_weight,
            max_weight,
        )

    out["decision_target_weight"] = out[config.output_col]
    out["held_target_weight"] = out[config.output_col]
    out["selection_side"] = np.select(
        [
            out["directional_state"].gt(0.0),
            out["directional_state"].lt(0.0),
        ],
        ["long", "short"],
        default="flat",
    )
    out["prior_target_weight"] = (
        out.groupby(config.product_col, sort=False)[config.output_col]
        .shift(1)
        .fillna(0.0)
    )
    out["target_turnover"] = (
        out[config.output_col] - out["prior_target_weight"]
    ).abs()

    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)
    daily = _residual_event_ttl_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_residual_event_ttl_v1",
        "entry_timing": "same_decision_row_age_1",
        "entry_threshold_inclusive": True,
        "decay_exit_threshold_inclusive": True,
        "same_direction_entry_action": "preserve_state_and_advance_age",
        "opposite_entry_action": "flip_and_reset_age",
        "missing_signal_action": "preserve_state_and_advance_age",
        "ttl_active_ages": f"1_through_{holding_periods}",
        "ttl_expiry_timing": "before_next_session_target",
        "normalization": "equal_weight_active_signs",
        "rescale_after_contract_cap": False,
        "additional_row_shift_periods": 0,
        "stop_rule": "not_applied_by_sleeve",
        "future_return_used": False,
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _validate_residual_event_ttl_contract(
    config: ExtractedSleeveConfig,
) -> None:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "expression": "directional",
        "construction": "residual_event_ttl",
        "normalization": "equal_weight_active_signs",
        "holding_unit": "sessions",
    }
    construction_geometry = str(
        parameters.get("construction_geometry") or ""
    ).strip().lower()
    if construction_geometry not in {
        "time_series_stateful",
        "cross_sectional_stateful",
    }:
        raise ValueError(
            f"{config.sleeve_id} residual-event adapter requires "
            "construction_geometry='time_series_stateful' or "
            "'cross_sectional_stateful'; received "
            f"{construction_geometry!r}"
        )
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} residual-event adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )

    try:
        entry_abs_z = float(parameters["entry_abs_z"])
        exit_abs_z = float(parameters["exit_abs_z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires numeric residual entry/exit thresholds"
        ) from exc
    if (
        not np.isfinite(entry_abs_z)
        or not np.isfinite(exit_abs_z)
        or entry_abs_z <= 0.0
        or exit_abs_z < 0.0
        or exit_abs_z >= entry_abs_z
    ):
        raise ValueError(
            f"{config.sleeve_id} requires 0 <= exit_abs_z < entry_abs_z"
        )

    try:
        holding_periods = int(parameters["holding_periods"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires positive integer holding_periods"
        ) from exc
    raw_holding_periods = parameters["holding_periods"]
    if (
        isinstance(raw_holding_periods, bool)
        or holding_periods < 1
        or float(raw_holding_periods) != float(holding_periods)
    ):
        raise ValueError(
            f"{config.sleeve_id} requires positive integer holding_periods"
        )

    lifecycle_defaults = {
        "entry_timing": "same_decision_row_age_1",
        "same_direction_entry_action": "preserve_state_and_advance_age",
        "opposite_entry_action": "flip_and_reset_age",
        "missing_signal_action": "preserve_state_and_advance_age",
        "ttl_expiry_timing": "before_next_session_target",
    }
    for field, required_value in lifecycle_defaults.items():
        actual = str(
            parameters.get(field, required_value)
        ).strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} residual-event adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )
    for field in ("entry_threshold_inclusive", "exit_threshold_inclusive"):
        if parameters.get(field, True) is not True:
            raise ValueError(
                f"{config.sleeve_id} residual-event adapter requires "
                f"{field}=True"
            )
    if parameters.get("stop_abs_z") is not None:
        raise ValueError(
            f"{config.sleeve_id} stop_abs_z must be implemented as a separate "
            "risk overlay, not inside the residual-event sleeve"
        )

    try:
        target_gross = float(parameters["target_gross_exposure"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires finite target_gross_exposure"
        ) from exc
    if not np.isfinite(target_gross) or not 0.0 < target_gross <= 2.0:
        raise ValueError(
            f"{config.sleeve_id} target_gross_exposure must be in (0, 2]"
        )

    max_weight = parameters.get("max_weight_per_contract")
    if max_weight is not None:
        try:
            max_weight = float(max_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be numeric"
            ) from exc
        if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be in (0, 1]"
            )
    if bool(parameters.get("rescale_after_contract_cap", False)):
        raise ValueError(
            f"{config.sleeve_id} residual-event adapter cannot rescale after "
            "the contract cap"
        )


def _residual_event_ttl_daily_summary(
    positions: pd.DataFrame,
    *,
    config: ExtractedSleeveConfig,
    target_gross: float,
) -> pd.DataFrame:
    output_col = config.output_col
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=(output_col, lambda values: float(values.abs().sum())),
        net_exposure=(output_col, "sum"),
        long_exposure=(
            output_col,
            lambda values: float(values.clip(lower=0.0).sum()),
        ),
        short_exposure=(
            output_col,
            lambda values: float((-values.clip(upper=0.0)).sum()),
        ),
        target_turnover=("target_turnover", "sum"),
        active_products=(
            "directional_state",
            lambda values: int(values.ne(0.0).sum()),
        ),
        long_products=(
            "directional_state",
            lambda values: int(values.gt(0.0).sum()),
        ),
        short_products=(
            "directional_state",
            lambda values: int(values.lt(0.0).sum()),
        ),
        state_entries=("state_entry_event", "sum"),
        state_flips=("state_flip_event", "sum"),
        decay_exits=("decay_exit_event", "sum"),
        ttl_exits=("ttl_exit_event", "sum"),
        same_direction_entries_ignored=(
            "same_direction_entry_ignored",
            "sum",
        ),
        states_preserved_on_missing=(
            "state_preserved_on_missing",
            "sum",
        ),
        missing_signals=("signal_missing", "sum"),
        contract_cap_count=("contract_cap_bound", "sum"),
        maximum_state_age=("state_age_periods", "max"),
    ).reset_index()
    daily["gross_realization"] = daily["gross_exposure"] / target_gross
    return daily


def _build_cross_sectional_z_tail(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Select inclusive score tails and equal-weight all active signed names.

    The factor already publishes an aligned cross-sectional z-score.  This
    adapter does not standardize or shift it again: every finite score whose
    absolute value reaches the frozen threshold is active on that decision
    row, and all active signs share one gross budget.  Long/short counts are
    not balanced, so net exposure may float.  Contract caps remain owned by
    the downstream strategy allocator.
    """

    if not (
        bool(frame.attrs.get("causal_signal_alignment_verified"))
        or bool(frame.attrs.get("causal_return_alignment_verified"))
    ):
        raise ExtractedSleeveAlignmentError(
            "extracted sleeve construction requires an upstream causal signal "
            "attestation"
        )
    _validate_cross_sectional_z_tail_contract(config)

    required = {
        config.date_col,
        config.product_col,
        *config.required_signal_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"extracted sleeve input is missing columns: {missing}"
        )

    source_attrs = dict(frame.attrs)
    out = frame.copy()
    out[config.date_col] = pd.to_datetime(
        out[config.date_col],
        errors="coerce",
    ).dt.normalize()
    out[config.product_col] = (
        out[config.product_col].astype("string").str.strip()
    )
    invalid_keys = (
        out[config.date_col].isna()
        | out[config.product_col].isna()
        | out[config.product_col].eq("")
    )
    if invalid_keys.any():
        raise ValueError(
            "extracted sleeve input contains invalid date/product keys"
        )
    if out.duplicated([config.date_col, config.product_col]).any():
        raise ValueError(
            "extracted sleeve input requires unique date/product rows"
        )

    out[config.signal_col] = pd.to_numeric(
        out[config.signal_col],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)

    parameters = dict(config.parameters or {})
    z_threshold = float(parameters["z_threshold"])
    target_gross = float(parameters["target_gross_exposure"])

    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_tail_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["tail_threshold_boundary"] = (
        oriented_signal.notna()
        & oriented_signal.abs().eq(z_threshold)
    )
    out["tail_selected"] = (
        oriented_signal.notna()
        & oriented_signal.abs().ge(z_threshold)
    )
    out["tail_direction"] = np.sign(oriented_signal).where(
        out["tail_selected"],
        0.0,
    )
    out["long_tail_selected"] = out["tail_direction"].gt(0.0)
    out["short_tail_selected"] = out["tail_direction"].lt(0.0)

    active_count = out["tail_selected"].groupby(
        out[config.date_col],
        sort=False,
    ).transform("sum")
    out["active_tail_count"] = active_count.astype(int)
    out["long_tail_count"] = out["long_tail_selected"].groupby(
        out[config.date_col],
        sort=False,
    ).transform("sum").astype(int)
    out["short_tail_count"] = out["short_tail_selected"].groupby(
        out[config.date_col],
        sort=False,
    ).transform("sum").astype(int)
    denominator = active_count.where(active_count.gt(0), np.nan)
    out["uncapped_target_weight"] = (
        out["tail_direction"] / denominator * target_gross
    ).fillna(0.0)
    out["contract_cap_bound"] = False
    out[config.output_col] = out["uncapped_target_weight"]
    out["decision_target_weight"] = out[config.output_col]
    out["held_target_weight"] = out[config.output_col]
    out["selection_side"] = np.select(
        [
            out["tail_direction"].gt(0.0),
            out["tail_direction"].lt(0.0),
        ],
        ["long", "short"],
        default="flat",
    )
    out["prior_target_weight"] = (
        out.groupby(config.product_col, sort=False)[config.output_col]
        .shift(1)
        .fillna(0.0)
    )
    out["target_turnover"] = (
        out[config.output_col] - out["prior_target_weight"]
    ).abs()

    daily = _cross_sectional_z_tail_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_cross_sectional_z_tail_v1",
        "threshold_operator": "inclusive_absolute_greater_than_or_equal",
        "normalization": "equal_weight_active_signs",
        "net_exposure_policy": "floating_from_tail_count_imbalance",
        "decision_refresh": "same_row_daily_no_state_carry",
        "additional_row_shift_periods": 0,
        "contract_cap_owner": "strategy_allocator",
        "sleeve_contract_cap": None,
        "future_return_used": False,
        "provenance": (
            "Recovered tail membership and all-active equal weighting; the "
            "historical generic 5% cap is intentionally external to this "
            "sleeve."
        ),
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _validate_cross_sectional_z_tail_contract(
    config: ExtractedSleeveConfig,
) -> None:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "construction_geometry": "cross_sectional",
        "expression": "directional",
        "construction": "cross_sectional_z_tail",
        "normalization": "equal_weight_active_signs",
        "holding_rule": "until_next_decision",
        "net_exposure_policy": "floating_from_tail_count_imbalance",
        "contract_cap_owner": "strategy_allocator",
    }
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} cross-sectional z-tail adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )

    try:
        z_threshold = float(parameters["z_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires a finite positive z_threshold"
        ) from exc
    if not np.isfinite(z_threshold) or z_threshold <= 0.0:
        raise ValueError(
            f"{config.sleeve_id} requires a finite positive z_threshold"
        )
    if parameters.get("threshold_inclusive") is not True:
        raise ValueError(
            f"{config.sleeve_id} requires threshold_inclusive=True"
        )
    for field in ("missing_signal_action", "non_tail_signal_action"):
        action = str(parameters.get(field) or "").strip().lower()
        if action != "flat":
            raise ValueError(
                f"{config.sleeve_id} requires {field}='flat'"
            )
    if parameters.get("state_carry") is not False:
        raise ValueError(
            f"{config.sleeve_id} requires state_carry=False"
        )
    if int(parameters.get("additional_row_shift_periods", -1)) != 0:
        raise ValueError(
            f"{config.sleeve_id} requires additional_row_shift_periods=0"
        )

    try:
        target_gross = float(parameters["target_gross_exposure"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires finite target_gross_exposure"
        ) from exc
    if not np.isfinite(target_gross) or not 0.0 < target_gross <= 2.0:
        raise ValueError(
            f"{config.sleeve_id} target_gross_exposure must be in (0, 2]"
        )

    if parameters.get("max_weight_per_contract") is not None:
        raise ValueError(
            f"{config.sleeve_id} max_weight_per_contract belongs to the "
            "strategy allocator, not the z-tail sleeve"
        )
    forbidden_risk_fields = (
        "max_sector_gross",
        "max_margin_utilization",
        "max_drawdown",
        "risk_overlay_id",
        "router_id",
        "stop_abs_z",
        "stop_loss",
        "volatility_target",
    )
    embedded = [
        field
        for field in forbidden_risk_fields
        if parameters.get(field) not in (None, "")
    ]
    if embedded:
        raise ValueError(
            f"{config.sleeve_id} embeds non-sleeve risk/router fields: "
            f"{embedded}; keep them in their owning components"
        )


def _cross_sectional_z_tail_daily_summary(
    positions: pd.DataFrame,
    *,
    config: ExtractedSleeveConfig,
    target_gross: float,
) -> pd.DataFrame:
    output_col = config.output_col
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=(output_col, lambda values: float(values.abs().sum())),
        net_exposure=(output_col, "sum"),
        long_exposure=(
            output_col,
            lambda values: float(values.clip(lower=0.0).sum()),
        ),
        short_exposure=(
            output_col,
            lambda values: float((-values.clip(upper=0.0)).sum()),
        ),
        target_turnover=("target_turnover", "sum"),
        active_products=("tail_selected", "sum"),
        long_products=("long_tail_selected", "sum"),
        short_products=("short_tail_selected", "sum"),
        missing_signals=("signal_missing", "sum"),
        threshold_boundary_selections=("tail_threshold_boundary", "sum"),
        contract_cap_count=("contract_cap_bound", "sum"),
    ).reset_index()
    daily["gross_realization"] = daily["gross_exposure"] / target_gross
    daily["one_sided_selection"] = (
        daily["long_products"].gt(0)
        ^ daily["short_products"].gt(0)
    )
    return daily


def _prepare_registered_extracted_input(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
    *,
    numeric_columns: tuple[str, ...],
) -> tuple[dict, pd.DataFrame]:
    """Validate and normalize a causal panel for a registered adapter."""

    if not (
        bool(frame.attrs.get("causal_signal_alignment_verified"))
        or bool(frame.attrs.get("causal_return_alignment_verified"))
    ):
        raise ExtractedSleeveAlignmentError(
            "extracted sleeve construction requires an upstream causal signal "
            "attestation"
        )
    required = {
        config.date_col,
        config.product_col,
        *config.required_signal_columns,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"extracted sleeve input is missing columns: {missing}"
        )

    source_attrs = dict(frame.attrs)
    out = frame.copy()
    out[config.date_col] = pd.to_datetime(
        out[config.date_col],
        errors="coerce",
    ).dt.normalize()
    out[config.product_col] = (
        out[config.product_col].astype("string").str.strip()
    )
    invalid_keys = (
        out[config.date_col].isna()
        | out[config.product_col].isna()
        | out[config.product_col].eq("")
    )
    if invalid_keys.any():
        raise ValueError(
            "extracted sleeve input contains invalid date/product keys"
        )
    if out.duplicated([config.date_col, config.product_col]).any():
        raise ValueError(
            "extracted sleeve input requires unique date/product rows"
        )
    for column in dict.fromkeys(numeric_columns):
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
    out = out.sort_values(
        [config.product_col, config.date_col],
        kind="mergesort",
    ).reset_index(drop=True)
    return source_attrs, out


def _attach_equal_weight_active_state_targets(
    out: pd.DataFrame,
    config: ExtractedSleeveConfig,
    *,
    target_gross: float,
    max_weight: float | None,
) -> None:
    """Equal-weight active signs, cap independently, and retain clipped cash."""

    active = out["directional_state"].ne(0.0)
    active_count = active.groupby(
        out[config.date_col],
        sort=False,
    ).transform("sum")
    out["active_state_count"] = active_count.astype(int)
    denominator = active_count.where(active_count.gt(0), np.nan)
    out["uncapped_target_weight"] = (
        out["directional_state"] / denominator * target_gross
    ).fillna(0.0)
    if max_weight is None:
        out["contract_cap_bound"] = False
        out[config.output_col] = out["uncapped_target_weight"]
    else:
        out["contract_cap_bound"] = out[
            "uncapped_target_weight"
        ].abs().gt(max_weight + 1e-15)
        out[config.output_col] = out["uncapped_target_weight"].clip(
            -max_weight,
            max_weight,
        )
    out["decision_target_weight"] = out[config.output_col]
    out["held_target_weight"] = out[config.output_col]
    out["selection_side"] = np.select(
        [
            out["directional_state"].gt(0.0),
            out["directional_state"].lt(0.0),
        ],
        ["long", "short"],
        default="flat",
    )
    out["prior_target_weight"] = (
        out.groupby(config.product_col, sort=False)[config.output_col]
        .shift(1)
        .fillna(0.0)
    )
    out["target_turnover"] = (
        out[config.output_col] - out["prior_target_weight"]
    ).abs()


def _registered_state_daily_summary(
    positions: pd.DataFrame,
    *,
    config: ExtractedSleeveConfig,
    target_gross: float,
    event_columns: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    output_col = config.output_col
    grouped = positions.groupby(config.date_col, sort=True)
    daily = grouped.agg(
        gross_exposure=(output_col, lambda values: float(values.abs().sum())),
        net_exposure=(output_col, "sum"),
        long_exposure=(
            output_col,
            lambda values: float(values.clip(lower=0.0).sum()),
        ),
        short_exposure=(
            output_col,
            lambda values: float((-values.clip(upper=0.0)).sum()),
        ),
        target_turnover=("target_turnover", "sum"),
        active_products=(
            "directional_state",
            lambda values: int(values.ne(0.0).sum()),
        ),
        long_products=(
            "directional_state",
            lambda values: int(values.gt(0.0).sum()),
        ),
        short_products=(
            "directional_state",
            lambda values: int(values.lt(0.0).sum()),
        ),
        missing_signals=("signal_missing", "sum"),
        neutral_signals=("signal_neutral", "sum"),
        contract_cap_count=("contract_cap_bound", "sum"),
        maximum_state_age=("state_age_periods", "max"),
    )
    for output_name, source_column in event_columns:
        daily[output_name] = grouped[source_column].sum()
    daily = daily.reset_index()
    daily["gross_realization"] = daily["gross_exposure"] / target_gross
    return daily


def _validate_equal_weight_state_sizing(
    config: ExtractedSleeveConfig,
) -> tuple[float, float | None]:
    parameters = dict(config.parameters or {})
    try:
        target_gross = float(parameters["target_gross_exposure"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires finite target_gross_exposure"
        ) from exc
    if not np.isfinite(target_gross) or not 0.0 < target_gross <= 2.0:
        raise ValueError(
            f"{config.sleeve_id} target_gross_exposure must be in (0, 2]"
        )
    max_weight_raw = parameters.get("max_weight_per_contract")
    max_weight = None
    if max_weight_raw is not None:
        try:
            max_weight = float(max_weight_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be numeric"
            ) from exc
        if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
            raise ValueError(
                f"{config.sleeve_id} max_weight_per_contract must be in (0, 1]"
            )
    if bool(parameters.get("rescale_after_contract_cap", False)):
        raise ValueError(
            f"{config.sleeve_id} cannot rescale after the contract cap"
        )
    if int(parameters.get("additional_row_shift_periods", -1)) != 0:
        raise ValueError(
            f"{config.sleeve_id} requires additional_row_shift_periods=0"
        )
    return target_gross, max_weight


def _validate_atr_donchian_exit_contract(
    config: ExtractedSleeveConfig,
) -> tuple[int, float, float | None]:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "construction_geometry": "time_series_stateful",
        "expression": "directional",
        "construction": "atr_donchian_exit_state",
        "normalization": "equal_weight_active_signs",
        "exit_price_col": "close",
        "exit_band_timing": "prior_sessions_shifted_one_row",
        "same_direction_entry_action": "preserve_state_and_advance_age",
        "opposite_entry_action": "flip_and_reset_age",
        "missing_signal_action": "preserve_state_unless_exit",
        "return_assumption": "close_signal_next_open_to_next_open",
    }
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} ATR-Donchian adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )
    try:
        exit_window = int(parameters["exit_window"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires integer exit_window"
        ) from exc
    raw_window = parameters["exit_window"]
    if (
        isinstance(raw_window, bool)
        or exit_window < 2
        or float(raw_window) != float(exit_window)
    ):
        raise ValueError(
            f"{config.sleeve_id} exit_window must be an integer >= 2"
        )
    expected_rule = (
        f"prior_{exit_window}_session_donchian_or_opposite_breakout"
    )
    if str(parameters.get("exit_rule") or "").strip().lower() != expected_rule:
        raise ValueError(
            f"{config.sleeve_id} exit_rule must equal {expected_rule!r}"
        )
    target_gross, max_weight = _validate_equal_weight_state_sizing(config)
    return exit_window, target_gross, max_weight


def _build_atr_donchian_exit_state(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Run a signed breakout lifecycle with a shifted prior Donchian exit."""

    exit_window, target_gross, max_weight = (
        _validate_atr_donchian_exit_contract(config)
    )
    source_attrs, out = _prepare_registered_extracted_input(
        frame,
        config,
        numeric_columns=(config.signal_col, "high", "low", "close"),
    )
    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_event_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["signal_neutral"] = oriented_signal.eq(0.0)
    out["state_update_event"] = oriented_signal.notna() & oriented_signal.ne(
        0.0
    )
    out["event_direction"] = np.sign(oriented_signal).where(
        out["state_update_event"],
        0.0,
    )
    out["donchian_prior_exit_high"] = out["high"].groupby(
        out[config.product_col],
        sort=False,
    ).transform(
        lambda values: values.shift(1).rolling(
            exit_window,
            min_periods=exit_window,
        ).max()
    )
    out["donchian_prior_exit_low"] = out["low"].groupby(
        out[config.product_col],
        sort=False,
    ).transform(
        lambda values: values.shift(1).rolling(
            exit_window,
            min_periods=exit_window,
        ).min()
    )

    row_count = len(out)
    directional_state = np.zeros(row_count, dtype=float)
    prior_directional_state = np.zeros(row_count, dtype=float)
    state_age_periods = np.zeros(row_count, dtype=int)
    prior_state_age_periods = np.zeros(row_count, dtype=int)
    state_entry_event = np.zeros(row_count, dtype=bool)
    state_flip_event = np.zeros(row_count, dtype=bool)
    donchian_exit_event = np.zeros(row_count, dtype=bool)
    same_direction_entry_ignored = np.zeros(row_count, dtype=bool)
    state_preserved_without_event = np.zeros(row_count, dtype=bool)
    lifecycle_exit_reason = np.full(row_count, "", dtype=object)

    for _, index in out.groupby(config.product_col, sort=False).groups.items():
        state = 0.0
        age = 0
        for position in index:
            prior_directional_state[position] = state
            prior_state_age_periods[position] = age
            event_direction = float(out["event_direction"].iat[position])
            close = out["close"].iat[position]
            prior_high = out["donchian_prior_exit_high"].iat[position]
            prior_low = out["donchian_prior_exit_low"].iat[position]
            long_exit = bool(
                state > 0.0
                and pd.notna(close)
                and pd.notna(prior_low)
                and close <= prior_low
            )
            short_exit = bool(
                state < 0.0
                and pd.notna(close)
                and pd.notna(prior_high)
                and close >= prior_high
            )

            if state == 0.0:
                if event_direction != 0.0:
                    state = event_direction
                    age = 1
                    state_entry_event[position] = True
            elif event_direction == -state:
                state = event_direction
                age = 1
                state_flip_event[position] = True
                lifecycle_exit_reason[position] = "opposite_entry_flip"
            elif long_exit or short_exit:
                state = 0.0
                age = 0
                donchian_exit_event[position] = True
                lifecycle_exit_reason[position] = "prior_donchian_extreme"
            else:
                same_direction_entry_ignored[position] = (
                    event_direction == state
                )
                state_preserved_without_event[position] = (
                    event_direction == 0.0
                )
                age += 1

            directional_state[position] = state
            state_age_periods[position] = age

    out["prior_directional_state"] = prior_directional_state
    out["prior_state_age_periods"] = prior_state_age_periods
    out["directional_state"] = directional_state
    out["state_age_periods"] = state_age_periods
    out["state_entry_event"] = state_entry_event
    out["state_flip_event"] = state_flip_event
    out["donchian_exit_event"] = donchian_exit_event
    out["same_direction_entry_ignored"] = same_direction_entry_ignored
    out["state_preserved_without_event"] = (
        state_preserved_without_event
    )
    out["lifecycle_exit_reason"] = lifecycle_exit_reason
    out["state_transition_event"] = out["directional_state"].ne(
        out["prior_directional_state"]
    )
    _attach_equal_weight_active_state_targets(
        out,
        config,
        target_gross=target_gross,
        max_weight=max_weight,
    )
    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)
    daily = _registered_state_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
        event_columns=(
            ("state_entries", "state_entry_event"),
            ("state_flips", "state_flip_event"),
            ("donchian_exits", "donchian_exit_event"),
            (
                "same_direction_entries_ignored",
                "same_direction_entry_ignored",
            ),
            ("states_preserved", "state_preserved_without_event"),
        ),
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_atr_donchian_exit_state_v1",
        "entry_timing": "same_decision_row",
        "exit_window": exit_window,
        "exit_band_timing": "prior_sessions_shifted_one_row",
        "exit_price": "current_decision_close",
        "same_direction_entry_action": "preserve_lifecycle",
        "opposite_entry_action": "flip_and_reset_lifecycle",
        "normalization": "equal_weight_active_signs",
        "rescale_after_contract_cap": False,
        "additional_row_shift_periods": 0,
        "future_return_used": False,
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _validate_atr_chandelier_exit_contract(
    config: ExtractedSleeveConfig,
) -> tuple[float, float, float | None]:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "construction_geometry": "time_series_stateful",
        "expression": "directional",
        "construction": "atr_chandelier_exit_state",
        "normalization": "equal_weight_active_signs",
        "entry_atr_col": "atr_breakout_atr",
        "entry_atr_policy": "freeze_at_entry",
        "exit_price_col": "close",
        "exit_extreme_timing": "prior_accumulated_extreme",
        "same_direction_entry_action": "preserve_state_and_entry_atr",
        "opposite_entry_action": "flip_and_reset_lifecycle",
        "missing_signal_action": "preserve_state_unless_exit",
        "return_assumption": "close_signal_next_open_to_next_open",
    }
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} ATR-Chandelier adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )
    try:
        atr_multiplier = float(parameters["atr_multiplier"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config.sleeve_id} requires finite positive atr_multiplier"
        ) from exc
    if not np.isfinite(atr_multiplier) or atr_multiplier <= 0.0:
        raise ValueError(
            f"{config.sleeve_id} requires finite positive atr_multiplier"
        )
    target_gross, max_weight = _validate_equal_weight_state_sizing(config)
    return atr_multiplier, target_gross, max_weight


def _build_atr_chandelier_exit_state(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Run a Chandelier lifecycle with entry-frozen ATR and prior extremes."""

    atr_multiplier, target_gross, max_weight = (
        _validate_atr_chandelier_exit_contract(config)
    )
    source_attrs, out = _prepare_registered_extracted_input(
        frame,
        config,
        numeric_columns=(
            config.signal_col,
            "high",
            "low",
            "close",
            "atr_breakout_atr",
        ),
    )
    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_event_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["signal_neutral"] = oriented_signal.eq(0.0)
    out["state_update_event"] = oriented_signal.notna() & oriented_signal.ne(
        0.0
    )
    out["event_direction"] = np.sign(oriented_signal).where(
        out["state_update_event"],
        0.0,
    )
    invalid_event_atr = (
        out["state_update_event"]
        & (
            out["atr_breakout_atr"].isna()
            | out["atr_breakout_atr"].le(0.0)
        )
    )
    if invalid_event_atr.any():
        raise ValueError(
            f"{config.sleeve_id} requires a finite positive "
            "atr_breakout_atr on every nonzero entry event"
        )

    row_count = len(out)
    directional_state = np.zeros(row_count, dtype=float)
    prior_directional_state = np.zeros(row_count, dtype=float)
    state_age_periods = np.zeros(row_count, dtype=int)
    prior_state_age_periods = np.zeros(row_count, dtype=int)
    frozen_entry_atr = np.full(row_count, np.nan, dtype=float)
    prior_frozen_entry_atr = np.full(row_count, np.nan, dtype=float)
    accumulated_extreme = np.full(row_count, np.nan, dtype=float)
    prior_accumulated_extreme = np.full(row_count, np.nan, dtype=float)
    chandelier_stop = np.full(row_count, np.nan, dtype=float)
    state_entry_event = np.zeros(row_count, dtype=bool)
    state_flip_event = np.zeros(row_count, dtype=bool)
    chandelier_exit_event = np.zeros(row_count, dtype=bool)
    same_direction_entry_ignored = np.zeros(row_count, dtype=bool)
    state_preserved_without_event = np.zeros(row_count, dtype=bool)
    lifecycle_exit_reason = np.full(row_count, "", dtype=object)

    for _, index in out.groupby(config.product_col, sort=False).groups.items():
        state = 0.0
        age = 0
        entry_atr = np.nan
        extreme = np.nan
        for position in index:
            prior_directional_state[position] = state
            prior_state_age_periods[position] = age
            prior_frozen_entry_atr[position] = entry_atr
            prior_accumulated_extreme[position] = extreme
            event_direction = float(out["event_direction"].iat[position])
            current_atr = out["atr_breakout_atr"].iat[position]
            high = out["high"].iat[position]
            low = out["low"].iat[position]
            close = out["close"].iat[position]

            if state > 0.0 and pd.notna(extreme) and pd.notna(entry_atr):
                chandelier_stop[position] = (
                    extreme - atr_multiplier * entry_atr
                )
            elif state < 0.0 and pd.notna(extreme) and pd.notna(entry_atr):
                chandelier_stop[position] = (
                    extreme + atr_multiplier * entry_atr
                )

            if state == 0.0:
                if event_direction != 0.0:
                    state = event_direction
                    age = 1
                    entry_atr = float(current_atr)
                    if state > 0.0:
                        extreme = (
                            float(high) if pd.notna(high) else float(close)
                        )
                    else:
                        extreme = (
                            float(low) if pd.notna(low) else float(close)
                        )
                    state_entry_event[position] = True
            elif event_direction == -state:
                state = event_direction
                age = 1
                entry_atr = float(current_atr)
                if state > 0.0:
                    extreme = (
                        float(high) if pd.notna(high) else float(close)
                    )
                else:
                    extreme = (
                        float(low) if pd.notna(low) else float(close)
                    )
                state_flip_event[position] = True
                lifecycle_exit_reason[position] = "opposite_entry_flip"
            else:
                stop = chandelier_stop[position]
                long_exit = bool(
                    state > 0.0
                    and pd.notna(close)
                    and pd.notna(stop)
                    and close <= stop
                )
                short_exit = bool(
                    state < 0.0
                    and pd.notna(close)
                    and pd.notna(stop)
                    and close >= stop
                )
                if long_exit or short_exit:
                    state = 0.0
                    age = 0
                    entry_atr = np.nan
                    extreme = np.nan
                    chandelier_exit_event[position] = True
                    lifecycle_exit_reason[position] = "prior_extreme_stop"
                else:
                    same_direction_entry_ignored[position] = (
                        event_direction == state
                    )
                    state_preserved_without_event[position] = (
                        event_direction == 0.0
                    )
                    age += 1
                    if state > 0.0 and pd.notna(high):
                        extreme = (
                            float(high)
                            if pd.isna(extreme)
                            else max(float(extreme), float(high))
                        )
                    elif state < 0.0 and pd.notna(low):
                        extreme = (
                            float(low)
                            if pd.isna(extreme)
                            else min(float(extreme), float(low))
                        )

            directional_state[position] = state
            state_age_periods[position] = age
            frozen_entry_atr[position] = entry_atr
            accumulated_extreme[position] = extreme

    out["prior_directional_state"] = prior_directional_state
    out["prior_state_age_periods"] = prior_state_age_periods
    out["directional_state"] = directional_state
    out["state_age_periods"] = state_age_periods
    out["prior_frozen_entry_atr"] = prior_frozen_entry_atr
    out["frozen_entry_atr"] = frozen_entry_atr
    out["prior_accumulated_extreme"] = prior_accumulated_extreme
    out["accumulated_extreme"] = accumulated_extreme
    out["chandelier_stop"] = chandelier_stop
    out["state_entry_event"] = state_entry_event
    out["state_flip_event"] = state_flip_event
    out["chandelier_exit_event"] = chandelier_exit_event
    out["same_direction_entry_ignored"] = same_direction_entry_ignored
    out["state_preserved_without_event"] = (
        state_preserved_without_event
    )
    out["lifecycle_exit_reason"] = lifecycle_exit_reason
    out["state_transition_event"] = out["directional_state"].ne(
        out["prior_directional_state"]
    )
    _attach_equal_weight_active_state_targets(
        out,
        config,
        target_gross=target_gross,
        max_weight=max_weight,
    )
    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)
    daily = _registered_state_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
        event_columns=(
            ("state_entries", "state_entry_event"),
            ("state_flips", "state_flip_event"),
            ("chandelier_exits", "chandelier_exit_event"),
            (
                "same_direction_entries_ignored",
                "same_direction_entry_ignored",
            ),
            ("states_preserved", "state_preserved_without_event"),
        ),
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_atr_chandelier_exit_state_v1",
        "entry_timing": "same_decision_row",
        "entry_atr_policy": "freeze_at_entry",
        "atr_multiplier": atr_multiplier,
        "exit_extreme_timing": "prior_accumulated_extreme",
        "same_direction_entry_action": "preserve_state_and_entry_atr",
        "opposite_entry_action": "flip_and_reset_lifecycle",
        "normalization": "equal_weight_active_signs",
        "rescale_after_contract_cap": False,
        "additional_row_shift_periods": 0,
        "future_return_used": False,
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


def _validate_fixed_event_hold_contract(
    config: ExtractedSleeveConfig,
) -> tuple[float, float | None]:
    if config.factor_id not in config.source_factor_ids:
        raise ValueError(
            f"{config.sleeve_id} is not registered for factor "
            f"{config.factor_id}"
        )
    parameters = dict(config.parameters or {})
    expected = {
        "construction_geometry": "cross_sectional",
        "expression": "directional",
        "construction": "fixed_event_hold",
        "normalization": "equal_weight_active_signs",
        "holding_unit": "sessions",
        "holding_rule": "one_decision_row_only",
        "zero_signal_action": "flat",
        "missing_signal_action": "flat",
        "return_assumption": "close_signal_next_open_to_close",
    }
    for field, required_value in expected.items():
        actual = str(parameters.get(field) or "").strip().lower()
        if actual != required_value:
            raise ValueError(
                f"{config.sleeve_id} fixed-event adapter requires "
                f"{field}={required_value!r}; received {actual!r}"
            )
    if parameters.get("state_carry") is not False:
        raise ValueError(
            f"{config.sleeve_id} fixed-event adapter requires state_carry=False"
        )
    raw_holding_periods = parameters.get("holding_periods")
    if (
        isinstance(raw_holding_periods, bool)
        or raw_holding_periods is None
        or float(raw_holding_periods) != 1.0
    ):
        raise ValueError(
            f"{config.sleeve_id} fixed-event adapter requires "
            "holding_periods=1"
        )
    return _validate_equal_weight_state_sizing(config)


def _build_fixed_event_hold(
    frame: pd.DataFrame,
    config: ExtractedSleeveConfig,
) -> ExtractedSleeveExecutionResult:
    """Map each nonzero event to one decision row with no state carry."""

    target_gross, max_weight = _validate_fixed_event_hold_contract(config)
    source_attrs, out = _prepare_registered_extracted_input(
        frame,
        config,
        numeric_columns=(config.signal_col,),
    )
    oriented_signal = out[config.signal_col].copy()
    if config.signal_orientation == "higher_is_bearish":
        oriented_signal = -oriented_signal
    out["sleeve_event_signal"] = oriented_signal
    out["signal_missing"] = oriented_signal.isna()
    out["signal_neutral"] = oriented_signal.eq(0.0)
    out["state_update_event"] = oriented_signal.notna() & oriented_signal.ne(
        0.0
    )
    out["directional_state"] = np.sign(oriented_signal).where(
        out["state_update_event"],
        0.0,
    )
    out["state_age_periods"] = out["directional_state"].ne(0.0).astype(int)
    out["prior_directional_state"] = (
        out.groupby(config.product_col, sort=False)["directional_state"]
        .shift(1)
        .fillna(0.0)
    )
    out["prior_state_age_periods"] = (
        out.groupby(config.product_col, sort=False)["state_age_periods"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    out["state_entry_event"] = out["directional_state"].ne(0.0)
    out["state_flat_event"] = out["directional_state"].eq(0.0)
    out["state_transition_event"] = out["directional_state"].ne(
        out["prior_directional_state"]
    )
    _attach_equal_weight_active_state_targets(
        out,
        config,
        target_gross=target_gross,
        max_weight=max_weight,
    )
    out = out.sort_values(
        [config.date_col, config.product_col],
        kind="mergesort",
    ).reset_index(drop=True)
    daily = _registered_state_daily_summary(
        out,
        config=config,
        target_gross=target_gross,
        event_columns=(
            ("active_events", "state_entry_event"),
            ("flat_rows", "state_flat_event"),
        ),
    )
    out.attrs.update(source_attrs)
    out.attrs["sleeve_id"] = config.sleeve_id
    out.attrs["sleeve_factor_id"] = config.factor_id
    out.attrs["sleeve_config"] = config.to_dict()
    out.attrs["sleeve_config_fingerprint"] = config.fingerprint
    out.attrs["extracted_sleeve_execution"] = {
        "schema_version": 1,
        "rule_family": config.rule_family,
        "adapter": "canonical_fixed_event_hold_v1",
        "holding_rule": "one_decision_row_only",
        "zero_signal_action": "flat",
        "missing_signal_action": "flat",
        "state_carry": False,
        "return_assumption": "close_signal_next_open_to_close",
        "normalization": "equal_weight_active_signs",
        "rescale_after_contract_cap": False,
        "additional_row_shift_periods": 0,
        "future_return_used": False,
    }
    return ExtractedSleeveExecutionResult(
        config=config,
        positions=out,
        daily_summary=daily,
    )


_EXTRACTED_SLEEVE_EXECUTION_ADAPTERS: dict[
    str,
    ExtractedSleeveAdapter,
] = {
    "atr_chandelier_exit_state": _build_atr_chandelier_exit_state,
    "atr_donchian_exit_state": _build_atr_donchian_exit_state,
    "cross_sectional_z_tail": _build_cross_sectional_z_tail,
    "fixed_event_hold": _build_fixed_event_hold,
    "opposite_event_state": _build_opposite_event_state,
    "residual_event_ttl": _build_residual_event_ttl,
}


__all__ = [
    "ExtractedSleeveAlignmentError",
    "ExtractedSleeveExecutionResult",
    "build_extracted_sleeve_targets",
    "supports_extracted_sleeve_execution",
]
