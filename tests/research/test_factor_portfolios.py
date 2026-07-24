from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from oqp.research.factor_portfolios import (
    FactorPortfolioComposer,
    FactorPortfolioConfig,
    FactorPortfolioRunner,
    FactorSpec,
    RouterSpec,
    SleeveSpec,
    compatible_router_inventory,
    contribution_summary,
    factor_correlation,
    factor_coverage,
    leave_one_out_summary,
    load_factor_portfolio_config,
    router_inventory,
)
from oqp.research.factor_portfolios.data import attach_instrument_classification
from oqp.research.strategy_routing import build_discrete_state_allocations


def _config(**overrides) -> FactorPortfolioConfig:
    values = {
        "strategy_id": "str_test_value_momentum",
        "name": "Test Value Momentum",
        "market_vertical": "FUTURES_CN",
        "factors": (
            FactorSpec("fac_value", weight=0.75),
            FactorSpec("fac_momentum", weight=0.25),
        ),
        "weighting_method": "static",
        "normalization": "cross_sectional_zscore",
        "missing_policy": "renormalize_available",
        "min_available_factors": 1,
        "max_weight_per_asset": 0.60,
    }
    values.update(overrides)
    return FactorPortfolioConfig(**values)


def _base_frame() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
        for ticker, close in zip(("A", "B", "C"), (100.0, 101.0, 102.0)):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "close": close,
                    "forward_return": 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "market_vertical": "FUTURES_CN",
            "data_frequency": "daily",
            "return_horizon": "1d",
            "execution_assumption": "custom_forward_return",
        }
    )
    return frame


def _factor_frame(values: list[float]) -> pd.DataFrame:
    frame = _base_frame()[["date", "ticker"]].copy()
    frame["factor_score"] = values
    frame.attrs.update(
        {
            "market_vertical": "FUTURES_CN",
            "data_frequency": "daily",
            "return_horizon": "1d",
            "evaluation_geometry": "cross_sectional",
            "execution_lag": "already_lagged",
            "return_assumption": "custom_forward_return",
            "alpha_signal_col": "factor_score",
        }
    )
    return frame


def test_static_weights_are_normalized_and_attributed() -> None:
    result = FactorPortfolioComposer(_config()).compose(
        _base_frame(),
        {
            "fac_value": _factor_frame([-1, 0, 1, -2, 0, 2]),
            "fac_momentum": _factor_frame([1, 0, -1, 2, 0, -2]),
        },
    )

    assert result.configured_weights == {"fac_value": 0.75, "fac_momentum": 0.25}
    contributions = list(result.contribution_columns.values())
    np.testing.assert_allclose(
        result.frame[contributions].sum(axis=1),
        result.frame["composite_score"],
    )
    assert set(result.frame["available_factor_count"]) == {2}
    assert set(result.frame["factor_weight_coverage"]) == {1.0}


def test_composer_excludes_ineligible_products_before_cross_sectional_ranking() -> None:
    base = _base_frame()
    base["liquidity_eligible"] = base["ticker"].ne("C")
    result = FactorPortfolioComposer(_config()).compose(
        base,
        {
            "fac_value": _factor_frame([-1, 0, 100, -2, 0, 200]),
            "fac_momentum": _factor_frame([1, 0, -100, 2, 0, -200]),
        },
    )

    excluded = result.frame.loc[result.frame["ticker"].eq("C")]
    assert excluded["composite_score"].isna().all()
    assert excluded["available_factor_count"].eq(0).all()


def test_missing_factor_is_renormalized_across_available_exposure() -> None:
    momentum = _factor_frame([1, 0, -1, np.nan, -1, 1])
    result = FactorPortfolioComposer(_config()).compose(
        _base_frame(),
        {
            "fac_value": _factor_frame([-1, 0, 1, -2, 0, 2]),
            "fac_momentum": momentum,
        },
    )
    target = result.frame.loc[
        (result.frame["date"] == pd.Timestamp("2025-01-03"))
        & (result.frame["ticker"] == "A")
    ].iloc[0]
    value_col = result.normalized_columns["fac_value"]
    assert target["available_factor_count"] == 1
    assert target["factor_weight_coverage"] == pytest.approx(0.75)
    assert target["composite_score"] == pytest.approx(target[value_col])


def test_cross_sectional_normalization_does_not_read_future_dates() -> None:
    original = _factor_frame([-1, 0, 1, -2, 0, 2])
    shocked = _factor_frame([-1, 0, 1, -2000, 0, 2000])
    other = _factor_frame([1, 0, -1, 1, 0, -1])
    composer = FactorPortfolioComposer(_config())
    first = composer.compose(
        _base_frame(), {"fac_value": original, "fac_momentum": other}
    )
    second = composer.compose(
        _base_frame(), {"fac_value": shocked, "fac_momentum": other}
    )
    first_date = first.frame["date"].eq(pd.Timestamp("2025-01-02"))
    np.testing.assert_allclose(
        first.frame.loc[first_date, "composite_score"],
        second.frame.loc[first_date, "composite_score"],
    )


def test_incompatible_factor_metadata_is_rejected() -> None:
    bad = _factor_frame([1, 0, -1, 1, 0, -1])
    bad.attrs["data_frequency"] = "intraday"
    with pytest.raises(ValueError, match="incompatible data_frequency"):
        FactorPortfolioComposer(_config()).compose(
            _base_frame(),
            {
                "fac_value": _factor_frame([-1, 0, 1, -2, 0, 2]),
                "fac_momentum": bad,
            },
        )


def test_diagnostics_use_audited_composition_columns() -> None:
    result = FactorPortfolioComposer(_config()).compose(
        _base_frame(),
        {
            "fac_value": _factor_frame([-1, 0, 1, -2, 0, 2]),
            "fac_momentum": _factor_frame([1, 0, -1, 2, 0, -2]),
        },
    )
    assert list(factor_correlation(result).columns) == ["fac_value", "fac_momentum"]
    assert set(factor_coverage(result)["factor_id"]) == {"fac_value", "fac_momentum"}
    assert set(contribution_summary(result)["factor_id"]) == {
        "fac_value",
        "fac_momentum",
    }
    leave_one_out = leave_one_out_summary(result)
    assert set(leave_one_out["omitted_factor"]) == {"fac_value", "fac_momentum"}
    assert leave_one_out["valid_rows"].eq(len(result.frame)).all()


def test_yaml_contract_and_equal_weights(tmp_path) -> None:
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        """
strategy_id: str_equal
name: Equal Blend
market_vertical: FUTURES_CN
factors:
  - factor_id: fac_value
    weight: 9
  - factor_id: fac_momentum
    weight: 1
blend:
  weighting_method: equal
  normalization: cross_sectional_rank
execution:
  mode: risk_desk
temporal:
  signal_frequency: session_close
  holding_mode: until_next_decision
success_criterion:
  profile_id: strategy_daily_internal_net_value_v1
""".strip(),
        encoding="utf-8",
    )
    config = load_factor_portfolio_config(config_path)
    assert config.normalized_weights == {"fac_value": 0.5, "fac_momentum": 0.5}
    assert config.normalization == "cross_sectional_rank"
    assert config.temporal_policy == {
        "signal_frequency": "session_close",
        "holding_mode": "until_next_decision",
    }
    assert (
        config.success_criterion_profile
        == "strategy_daily_internal_net_value_v1"
    )


def test_runner_reuses_execution_mode_after_composition() -> None:
    contract = {
        "evaluation_geometry": "cross_sectional",
        "execution_mode": "risk_desk",
        "alpha_signal_col": "factor_score",
        "execution_weight_col": "factor_score",
        "execution_lag": "already_lagged",
        "return_assumption": "custom_forward_return",
        "supported_markets": ["FUTURES_CN"],
    }

    def module_for(multiplier: float) -> SimpleNamespace:
        def compute(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            ticker_score = out["ticker"].map({"A": -1.0, "B": 0.0, "C": 1.0})
            out["factor_score"] = ticker_score * multiplier
            return out

        return SimpleNamespace(
            FACTOR_ID="synthetic",
            FACTOR_CONTRACT=contract,
            FACTOR_METADATA={"supported_markets": ["FUTURES_CN"]},
            compute=compute,
        )

    modules = {
        "fac_value": module_for(1.0),
        "fac_momentum": module_for(-1.0),
    }
    with patch(
        "oqp.research.factor_portfolios.runner.load_factor_module",
        side_effect=lambda factor_id: modules[factor_id],
    ):
        result = FactorPortfolioRunner(
            _config(
                success_criterion_profile=(
                    "strategy_daily_internal_net_value_v1"
                )
            )
        ).build(_base_frame())

    assert "final_target_weight" in result.frame.columns
    assert result.frame.attrs["strategy_id"] == "str_test_value_momentum"
    assert result.frame.attrs["factor_contract"]["contract_source"] == "factor_portfolio"
    assert result.frame.attrs["factor_params"]["component_type"] == "factor_portfolio"
    assert result.frame["signal_decision_row"].all()
    assert result.frame.attrs["temporal_policy"]["holding_mode"] == "until_next_decision"
    assert (
        result.frame.attrs["success_criterion_profile_id"]
        == "strategy_daily_internal_net_value_v1"
    )
    assert len(result.frame.attrs["success_criterion_fingerprint"]) == 64
    assert result.frame.attrs["success_criterion_status"] == "declared_not_evaluated"


def test_runner_routes_independently_constructed_sleeves() -> None:
    contract = {
        "evaluation_geometry": "cross_sectional",
        "execution_mode": "risk_desk",
        "alpha_signal_col": "factor_score",
        "execution_weight_col": "factor_score",
        "execution_lag": "already_lagged",
        "return_assumption": "custom_forward_return",
        "supported_markets": ["FUTURES_CN"],
    }

    def module_for(multiplier: float) -> SimpleNamespace:
        def compute(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out["factor_score"] = out["ticker"].map(
                {"A": -1.0, "B": 0.0, "C": 1.0}
            ) * multiplier
            return out

        return SimpleNamespace(
            FACTOR_ID="synthetic",
            FACTOR_CONTRACT=contract,
            FACTOR_METADATA={"supported_markets": ["FUTURES_CN"]},
            compute=compute,
        )

    def route(states, *, sleeve_ids, parameters):
        return build_discrete_state_allocations(
            states,
            parameters["assignments"],
            sleeve_ids=sleeve_ids,
            state_col="state",
            decision_date_col="decision_date",
            effective_date_col="effective_date",
        )

    router_module = SimpleNamespace(
        ROUTER_ID="rtr_test_switch",
        ROUTER_CONTRACT={
            "decision_lag_periods": 1,
            "supported_markets": ["FUTURES_CN"],
        },
        route=route,
    )
    config = FactorPortfolioConfig(
        strategy_id="str_test_router",
        name="Test Router",
        market_vertical="FUTURES_CN",
        sleeves=(
            SleeveSpec("value", (FactorSpec("fac_value"),)),
            SleeveSpec("momentum", (FactorSpec("fac_momentum"),)),
        ),
        router=RouterSpec(
            "rtr_test_switch",
            parameters={"assignments": {"value": "value", "momentum": "momentum"}},
        ),
        max_weight_per_asset=0.60,
    )
    states = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "effective_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "state": ["value", "momentum"],
        }
    )
    modules = {
        "fac_value": module_for(1.0),
        "fac_momentum": module_for(-1.0),
    }
    with (
        patch(
            "oqp.research.factor_portfolios.runner.load_factor_module",
            side_effect=lambda factor_id: modules[factor_id],
        ),
        patch(
            "oqp.research.factor_portfolios.runner.load_router_module",
            return_value=router_module,
        ),
    ):
        result = FactorPortfolioRunner(config).build(
            _base_frame(), router_states=states
        )

    first = result.frame.loc[result.frame["date"].eq(pd.Timestamp("2025-01-02"))]
    second = result.frame.loc[result.frame["date"].eq(pd.Timestamp("2025-01-03"))]
    assert first.set_index("ticker").loc["A", "final_target_weight"] < 0
    assert second.set_index("ticker").loc["A", "final_target_weight"] > 0
    assert result.router_result is not None
    assert result.composition is None


def test_instrument_master_supplies_missing_futures_sector() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": ["黄金(au)[指数]", "TA609"],
            "close": [500.0, 6000.0],
        }
    )
    enriched = attach_instrument_classification(frame, "FUTURES_CN")
    assert enriched["sector"].tolist() == ["贵金属", "化工"]


def test_router_inventory_is_separate_and_market_filterable() -> None:
    module = SimpleNamespace(
        __name__="synthetic_router",
        ROUTER_ID="rtr_test",
        ROUTER_METADATA={
            "name": "Test Router",
            "status": "diagnostic",
            "frequency": "monthly",
        },
        ROUTER_CONTRACT={"supported_markets": ["FUTURES_CN"]},
    )
    with (
        patch(
            "oqp.research.factor_portfolios.inventory.iter_router_files",
            return_value=(Path("/tmp/rtr_test.py"),),
        ),
        patch(
            "oqp.research.factor_portfolios.inventory.load_router_module",
            return_value=module,
        ),
    ):
        inventory = router_inventory()

    assert inventory["router_id"].tolist() == ["rtr_test"]
    assert len(compatible_router_inventory(inventory, "FUTURES_CN")) == 1
    assert compatible_router_inventory(inventory, "EQUITY_US").empty
