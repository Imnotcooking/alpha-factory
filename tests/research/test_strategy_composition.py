from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from oqp.research.strategy_composition import (
    FrozenRouterComponent,
    FrozenSleeveComponent,
    StrategyAllocatorConfig,
    StrategyCompositionConfig,
    StrategyExecutionConfig,
    audit_strategy_composition_readiness,
    compose_strategy,
    load_strategy_composition_config,
)
from oqp.research.strategy_routing import RouterContract


def _positions(targets: list[float], *, fake_cost: float = 0.0) -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "A"],
            "close": [100.0, 101.0],
            "research_split": ["validation", "holdout"],
            "forward_return": [0.01, -0.01],
            "next_symbol": ["A2501", "A2501"],
            "next_actual_open": [100.0, 101.0],
            "next_multiplier": [10.0, 10.0],
            "next_tick_size": [1.0, 1.0],
            "next_fee_type": ["fixed", "fixed"],
            "next_fee_open": [2.0, 2.0],
            "next_fee_close_today": [3.0, 3.0],
            "target_weight": targets,
            "cost_return": [fake_cost, fake_cost],
            "net_contribution": [-fake_cost, -fake_cost],
        }
    )


def _sleeve(sleeve_id: str, targets: list[float], *, fake_cost: float = 0.0):
    return FrozenSleeveComponent(
        sleeve_id=sleeve_id,
        factor_id=f"fac_{sleeve_id}",
        market_vertical="FUTURES_CN",
        return_assumption="close_signal_next_open_to_close",
        config_fingerprint=f"fingerprint-{sleeve_id}",
        standalone_status="eligible_for_router_review",
        router_eligible=True,
        target_positions=_positions(targets, fake_cost=fake_cost),
    )


def _router(
    sleeve_a: str,
    sleeve_b: str,
    *,
    weight_a: float = 1.0,
) -> FrozenRouterComponent:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    "decision_date": date,
                    "effective_date": date,
                    "sleeve_id": sleeve_a,
                    "allocation": weight_a,
                },
                {
                    "decision_date": date,
                    "effective_date": date,
                    "sleeve_id": sleeve_b,
                    "allocation": 1.0 - weight_a,
                },
            ]
        )
    return FrozenRouterComponent(
        router_id="rtr_test",
        market_vertical="FUTURES_CN",
        sleeve_ids=(sleeve_a, sleeve_b),
        evidence_fingerprint="router-fingerprint",
        router_status="eligible_for_strategy_review",
        contract=RouterContract(
            router_id="rtr_test",
            decision_lag_periods=0,
            supported_markets=("FUTURES_CN",),
        ),
        allocations=pd.DataFrame(rows),
    )


def _config(*, overlays: tuple[str, ...] = ()) -> StrategyCompositionConfig:
    return StrategyCompositionConfig(
        strategy_id="str_test",
        name="Test Composition",
        market_vertical="FUTURES_CN",
        sleeves=("slv_trend", "slv_reversal"),
        router="rtr_test",
        risk_overlays=overlays,
        allocator=StrategyAllocatorConfig(
            max_gross_leverage=1.0, max_contract_weight=0.25
        ),
        execution=StrategyExecutionConfig(
            capital=10_000_000,
            capital_currency="CNY",
            transaction_cost_profile="cn_futures_broker_v1",
            slippage_ticks_per_side=0.5,
        ),
    )


def test_phase7_yaml_loads_only_stable_component_references(tmp_path: Path) -> None:
    path = tmp_path / "strategy.yaml"
    path.write_text(
        """
strategy:
  strategy_id: str_cn_futures_router
  name: CN futures routed strategy
  market_vertical: FUTURES_CN
  sleeves:
    - slv_trend
    - slv_reversal
  router: rtr_volatility_threshold
  risk_overlays:
    - ovl_drawdown_brake
  allocator:
    max_gross_leverage: 1.0
    max_contract_weight: 0.10
  execution:
    capital: 10000000
    capital_currency: CNY
    transaction_cost_profile: cn_futures_broker_v1
    slippage_ticks_per_side: 0.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config = load_strategy_composition_config(path)
    assert config.sleeves == ("slv_trend", "slv_reversal")
    assert config.router == "rtr_volatility_threshold"
    assert config.risk_overlays == ("ovl_drawdown_brake",)
    assert len(config.fingerprint) == 64


def test_phase7_yaml_rejects_embedded_component_parameters() -> None:
    with pytest.raises(ValueError, match="list of stable slv"):
        StrategyCompositionConfig.from_mapping(
            {
                "strategy": {
                    "strategy_id": "str_bad",
                    "name": "Bad",
                    "market_vertical": "FUTURES_CN",
                    "sleeves": [
                        {"sleeve_id": "slv_trend", "lookback": 20},
                        "slv_reversal",
                    ],
                    "router": "rtr_test",
                }
            }
        )


def test_phase7_rejects_unvalidated_components() -> None:
    sleeves = {
        "slv_trend": _sleeve("slv_trend", [0.5, 0.5]),
        "slv_reversal": _sleeve("slv_reversal", [-0.5, -0.5]),
    }
    sleeves["slv_reversal"] = replace(
        sleeves["slv_reversal"], router_eligible=False
    )
    with pytest.raises(ValueError, match="Phase 4 standalone gate"):
        compose_strategy(
            _config(), sleeves, router=_router("slv_trend", "slv_reversal")
        )


def test_router_then_overlay_then_allocator_and_inputs_remain_immutable() -> None:
    def apply(targets: pd.DataFrame, *, parameters) -> pd.DataFrame:
        out = targets.copy()
        out["final_target_weight"] *= 0.5
        return out

    overlay = SimpleNamespace(
        __name__="ovl_test",
        OVERLAY_ID="ovl_test",
        OVERLAY_CONTRACT={
            "supported_markets": ["FUTURES_CN"],
            "source_weight_col": "final_target_weight",
            "output_weight_col": "final_target_weight",
        },
        DEFAULT_PARAMETERS={},
        apply=apply,
    )
    sleeves = {
        "slv_trend": _sleeve("slv_trend", [0.8, 0.8]),
        "slv_reversal": _sleeve("slv_reversal", [-0.2, -0.2]),
    }
    originals = {
        key: value.target_positions.copy(deep=True) for key, value in sleeves.items()
    }
    bundle = compose_strategy(
        _config(overlays=("ovl_test",)),
        sleeves,
        router=_router("slv_trend", "slv_reversal", weight_a=1.0),
        overlay_modules={"ovl_test": overlay},
    )
    assert bundle.transformation_audit["step"].tolist() == [
        "router_allocation",
        "risk_overlay:ovl_test",
        "allocator",
    ]
    assert bundle.positions["routed_target_weight"].tolist() == [0.8, 0.8]
    assert bundle.positions["final_target_weight"].tolist() == [0.4, 0.4]
    assert bundle.positions["allocated_target_weight"].tolist() == [0.25, 0.25]
    for key, original in originals.items():
        pd.testing.assert_frame_equal(sleeves[key].target_positions, original)


def test_final_costs_ignore_sleeve_costs_and_charge_only_netted_positions() -> None:
    sleeves = {
        "slv_trend": _sleeve("slv_trend", [0.5, 0.5], fake_cost=0.99),
        "slv_reversal": _sleeve(
            "slv_reversal", [-0.5, -0.5], fake_cost=0.77
        ),
    }
    bundle = compose_strategy(
        _config(),
        sleeves,
        router=_router("slv_trend", "slv_reversal", weight_a=0.5),
    )
    assert bundle.positions["allocated_target_weight"].eq(0.0).all()
    assert bundle.positions["contracts"].eq(0.0).all()
    assert bundle.daily_returns["cost_return"].eq(0.0).all()
    assert bundle.manifest["sleeve_hypothetical_cost_columns_ignored"] is True
    assert bundle.manifest["costs_computed_once_after_final_position_changes"] is True


def test_cn_futures_composition_enforces_margin_budget_after_routing() -> None:
    sleeves = {
        "slv_trend": _sleeve("slv_trend", [0.8, 0.8]),
        "slv_reversal": _sleeve("slv_reversal", [-0.2, -0.2]),
    }
    config = replace(
        _config(),
        allocator=StrategyAllocatorConfig(
            max_gross_leverage=None,
            max_contract_weight=1.0,
            max_margin_utilization=0.05,
        ),
    )
    bundle = compose_strategy(
        config,
        sleeves,
        router=_router("slv_trend", "slv_reversal", weight_a=1.0),
    )

    assert bundle.positions["margin_cap_bound"].all()
    assert bundle.positions["allocated_target_weight"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert bundle.daily_returns["target_margin_utilization"].tolist() == (
        pytest.approx([0.05, 0.05])
    )


def test_current_phase7_readiness_is_blocked() -> None:
    root = Path(__file__).resolve().parents[2]
    summary, components = audit_strategy_composition_readiness(
        root / "runtime/artifacts/research/router_hypotheses",
        root / "departments/research/strategies/compositions",
    )
    assert summary["status"] == "blocked"
    assert summary["eligible_phase6_routers"] == 0
    assert list(components.columns) == [
        "component_type",
        "component_id",
        "status",
        "eligible",
        "reason",
        "artifact_path",
    ]
