"""Shared state-map allocation and routed-position construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from oqp.research.strategy_routing.contracts import RouterContract


@dataclass(frozen=True)
class RoutedSleeveResult:
    frame: pd.DataFrame
    allocations: pd.DataFrame
    contributions: pd.DataFrame


def build_discrete_state_allocations(
    states: pd.DataFrame,
    assignments: Mapping[str, str],
    *,
    sleeve_ids: Sequence[str],
    state_col: str,
    decision_date_col: str,
    effective_date_col: str,
    period_frequency: str | None = None,
    flat_label: str = "flat",
) -> pd.DataFrame:
    """Map each causal state observation to one fully allocated sleeve."""

    required = {state_col, decision_date_col, effective_date_col}
    missing = sorted(required.difference(states.columns))
    if missing:
        raise ValueError(f"router state frame is missing columns: {missing}")
    allowed_sleeves = set(map(str, sleeve_ids))
    normalized_assignments = {str(key): str(value) for key, value in assignments.items()}
    unknown = sorted(set(normalized_assignments.values()).difference(allowed_sleeves | {flat_label}))
    if unknown:
        raise ValueError(f"router assignments reference unknown sleeves: {unknown}")

    out = states[[decision_date_col, effective_date_col, state_col]].copy()
    out[state_col] = out[state_col].astype(str)
    unmapped = sorted(set(out[state_col]).difference(normalized_assignments))
    if unmapped:
        raise ValueError(f"router states have no assignment: {unmapped}")
    if out[effective_date_col].duplicated().any():
        raise ValueError("router state frame has duplicate effective dates")

    if period_frequency:
        decision = out[decision_date_col].map(
            lambda value: pd.Period(value, freq=period_frequency)
        )
        effective = out[effective_date_col].map(
            lambda value: pd.Period(value, freq=period_frequency)
        )
        if not (decision + 1).equals(effective):
            raise ValueError("router state timing must apply period t to period t+1")

    out["selected_sleeve"] = out[state_col].map(normalized_assignments)
    out["_router_cross_key"] = 1
    sleeve_frame = pd.DataFrame(
        {"sleeve_id": list(map(str, sleeve_ids)), "_router_cross_key": 1}
    )
    out = out.merge(sleeve_frame, on="_router_cross_key", how="inner").drop(
        columns="_router_cross_key"
    )
    out["allocation"] = out["sleeve_id"].eq(out["selected_sleeve"]).astype(float)
    out = out.rename(
        columns={
            decision_date_col: "decision_date",
            effective_date_col: "effective_date",
            state_col: "state",
        }
    )
    return out.reset_index(drop=True)


def validate_router_allocations(
    allocations: pd.DataFrame,
    contract: RouterContract,
    *,
    sleeve_ids: Sequence[str],
) -> pd.DataFrame:
    required = {
        contract.decision_date_col,
        contract.effective_date_col,
        contract.sleeve_col,
        contract.allocation_col,
    }
    missing = sorted(required.difference(allocations.columns))
    if missing:
        raise ValueError(f"router allocations are missing columns: {missing}")
    out = allocations.copy()
    out[contract.decision_date_col] = pd.to_datetime(
        out[contract.decision_date_col], errors="raise"
    )
    out[contract.effective_date_col] = pd.to_datetime(
        out[contract.effective_date_col], errors="raise"
    )
    if contract.decision_lag_periods > 0 and not out[contract.effective_date_col].gt(
        out[contract.decision_date_col]
    ).all():
        raise ValueError("router effective dates must follow decision dates")
    out[contract.sleeve_col] = out[contract.sleeve_col].astype(str)
    unknown = sorted(set(out[contract.sleeve_col]).difference(map(str, sleeve_ids)))
    if unknown:
        raise ValueError(f"router allocations reference unknown sleeves: {unknown}")
    out[contract.allocation_col] = pd.to_numeric(
        out[contract.allocation_col], errors="raise"
    )
    if not contract.allow_negative_allocation and out[contract.allocation_col].lt(0.0).any():
        raise ValueError("router allocations cannot be negative")
    if out.duplicated([contract.effective_date_col, contract.sleeve_col]).any():
        raise ValueError("router allocations must be unique by effective date and sleeve")
    gross = out.groupby(contract.effective_date_col, observed=True)[
        contract.allocation_col
    ].apply(lambda values: values.abs().sum())
    if gross.gt(1.0 + 1e-12).any():
        raise ValueError("router sleeve allocations exceed 100% gross capital")
    if not contract.allow_partial_allocation and not np.allclose(gross, 1.0):
        raise ValueError("router sleeve allocations must sum to 100% on each decision")
    return out.sort_values(
        [contract.effective_date_col, contract.sleeve_col]
    ).reset_index(drop=True)


def route_sleeve_targets(
    base_frame: pd.DataFrame,
    sleeve_targets: Mapping[str, pd.DataFrame],
    allocations: pd.DataFrame,
    contract: RouterContract,
) -> RoutedSleeveResult:
    """Apply lagged sleeve allocations and aggregate only final target positions."""

    if not sleeve_targets:
        raise ValueError("router requires at least one sleeve target frame")
    required_base = {"date", "ticker"}
    if missing := sorted(required_base.difference(base_frame.columns)):
        raise ValueError(f"base frame is missing columns: {missing}")
    sleeve_ids = tuple(sleeve_targets)
    validated = validate_router_allocations(
        allocations, contract, sleeve_ids=sleeve_ids
    )
    effective_col = contract.effective_date_col
    sleeve_col = contract.sleeve_col
    allocation_col = contract.allocation_col

    contributions: list[pd.DataFrame] = []
    for sleeve_id, targets in sleeve_targets.items():
        required = {"date", "ticker", "target_weight"}
        if missing := sorted(required.difference(targets.columns)):
            raise ValueError(f"{sleeve_id} targets are missing columns: {missing}")
        target = targets[["date", "ticker", "target_weight"]].copy()
        target["date"] = pd.to_datetime(target["date"], errors="raise")
        target["target_weight"] = pd.to_numeric(
            target["target_weight"], errors="coerce"
        ).fillna(0.0)
        schedule = validated.loc[
            validated[sleeve_col].eq(sleeve_id),
            [effective_col, allocation_col],
        ].sort_values(effective_col)
        if schedule.empty:
            target["sleeve_allocation"] = 0.0
        else:
            target = pd.merge_asof(
                target.sort_values("date"),
                schedule.rename(columns={effective_col: "date"}),
                on="date",
                direction="backward",
            )
            target["sleeve_allocation"] = target[allocation_col].fillna(0.0)
            target = target.drop(columns=[allocation_col])
        target["sleeve_id"] = sleeve_id
        target["routed_contribution"] = (
            target["target_weight"] * target["sleeve_allocation"]
        )
        contributions.append(target)

    contribution_frame = pd.concat(contributions, ignore_index=True)
    routed = (
        contribution_frame.groupby(["date", "ticker"], observed=True)[
            "routed_contribution"
        ]
        .sum()
        .rename("routed_target_weight")
        .reset_index()
    )
    frame = base_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.merge(routed, on=["date", "ticker"], how="left", validate="one_to_one")
    frame["routed_target_weight"] = frame["routed_target_weight"].fillna(0.0)
    frame.attrs.update(base_frame.attrs)
    frame.attrs["router_id"] = contract.router_id
    frame.attrs["router_contract"] = contract.to_dict()
    return RoutedSleeveResult(
        frame=frame,
        allocations=validated,
        contributions=contribution_frame,
    )


__all__ = [
    "RoutedSleeveResult",
    "build_discrete_state_allocations",
    "route_sleeve_targets",
    "validate_router_allocations",
]
