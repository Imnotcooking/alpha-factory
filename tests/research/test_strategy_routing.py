from __future__ import annotations

import pandas as pd
import pytest

from oqp.research.strategy_routing import (
    RouterContract,
    build_discrete_state_allocations,
    iter_router_files,
    load_router_module,
    route_sleeve_targets,
    validate_router_allocations,
)
from oqp.optimization import resolve_component_parameter_schema


def _states() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_month": ["2024-12", "2025-01"],
            "holding_month": ["2025-01", "2025-02"],
            "volatility_state": ["Q1", "Q4"],
        }
    )


def _contract(*, allow_partial: bool = False) -> RouterContract:
    return RouterContract(
        router_id="rtr_test",
        allow_partial_allocation=allow_partial,
        supported_markets=("FUTURES_CN",),
    )


def test_discrete_state_router_is_lagged_and_fully_auditable() -> None:
    allocations = build_discrete_state_allocations(
        _states(),
        {"Q1": "trend", "Q4": "reversal"},
        sleeve_ids=("trend", "reversal"),
        state_col="volatility_state",
        decision_date_col="signal_month",
        effective_date_col="holding_month",
        period_frequency="M",
    )
    validated = validate_router_allocations(
        allocations,
        _contract(),
        sleeve_ids=("trend", "reversal"),
    )
    assert len(validated) == 4
    selected = validated.loc[validated["allocation"].eq(1.0), "sleeve_id"]
    assert selected.tolist() == ["trend", "reversal"]


def test_discrete_state_router_rejects_noncausal_month_alignment() -> None:
    states = _states()
    states.loc[0, "holding_month"] = "2024-12"
    with pytest.raises(ValueError, match=r"period t to period t\+1"):
        build_discrete_state_allocations(
            states,
            {"Q1": "trend", "Q4": "reversal"},
            sleeve_ids=("trend", "reversal"),
            state_col="volatility_state",
            decision_date_col="signal_month",
            effective_date_col="holding_month",
            period_frequency="M",
        )


def test_final_positions_are_formed_after_router_allocation() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03"])
    base = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A", "A"],
            "close": [100.0, 101.0, 102.0],
        }
    )
    trend = base[["date", "ticker"]].assign(target_weight=[0.5, 0.5, 0.5])
    reversal = base[["date", "ticker"]].assign(target_weight=[-0.25, -0.25, -0.25])
    allocations = build_discrete_state_allocations(
        _states(),
        {"Q1": "trend", "Q4": "reversal"},
        sleeve_ids=("trend", "reversal"),
        state_col="volatility_state",
        decision_date_col="signal_month",
        effective_date_col="holding_month",
        period_frequency="M",
    )
    result = route_sleeve_targets(
        base,
        {"trend": trend, "reversal": reversal},
        allocations,
        _contract(),
    )
    assert result.frame["routed_target_weight"].tolist() == [0.5, 0.5, -0.25]
    assert "routed_contribution" in result.contributions.columns


def test_reproducible_routers_expose_valid_declarative_parameters() -> None:
    router_files = iter_router_files()
    assert router_files

    for path in router_files:
        module = load_router_module(path.stem)
        schema = resolve_component_parameter_schema(
            module, component_type="router"
        )
        assert schema.component_id == module.ROUTER_ID
        assert len(schema.fingerprint) == 64
