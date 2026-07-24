from __future__ import annotations

import math

import pandas as pd
import pytest

from oqp.data.instruments import InstrumentMaster
from oqp.execution.transaction_costs import (
    CostUseCase,
    TransactionCostReadinessError,
    TransactionCostRegistry,
    attach_transaction_cost_policy,
)


def test_registry_routes_each_supported_market_to_an_explicit_profile() -> None:
    registry = TransactionCostRegistry.load()

    assert registry.resolve("FUTURES_CN").profile_id == "cn_futures_broker_v1"
    assert registry.resolve("EQUITY_US").profile_id == "ibkr_pro_fixed_us_equity_v1"
    assert registry.resolve("OPTIONS_US").profile_id == "ibkr_pro_us_options_v1"
    assert registry.resolve("FUTURES_US").profile_id == "us_futures_broker_pending"


def test_us_futures_placeholder_allows_only_explicit_gross_research() -> None:
    profile = TransactionCostRegistry.load().resolve("FUTURES_US")

    profile.assert_ready(CostUseCase.EXPLORATORY_GROSS)
    with pytest.raises(TransactionCostReadinessError, match="placeholder"):
        profile.assert_ready(CostUseCase.RESEARCH_NET)
    with pytest.raises(TransactionCostReadinessError, match="placeholder"):
        profile.assert_ready(CostUseCase.PRODUCTION)


def test_us_futures_never_falls_through_to_chinese_contract_specs() -> None:
    master = InstrumentMaster("FUTURES_US")

    with pytest.raises(ValueError, match="FUTURES_US"):
        master.get_profile("ES")


def test_ibkr_fixed_us_equity_commission_applies_order_floor_and_value_cap() -> None:
    registry = TransactionCostRegistry.load()
    profile = registry.resolve("EQUITY_US")

    ordinary = registry.estimate_order_cost(profile, quantity=100, price=25.0, side="buy")
    low_price = registry.estimate_order_cost(profile, quantity=1_000, price=0.25, side="buy")

    assert math.isclose(ordinary.broker_commission, 1.0)
    assert math.isclose(low_price.broker_commission, 2.5)
    assert math.isclose(ordinary.regulatory_fees, 100 * 0.000003)
    assert "portfolio TCA inputs" in ordinary.omissions[0]


def test_ibkr_fixed_us_equity_sell_adds_sec_and_finra_fees() -> None:
    registry = TransactionCostRegistry.load()
    estimate = registry.estimate_order_cost(
        registry.resolve("EQUITY_US"),
        quantity=100,
        price=25.0,
        side="sell",
    )

    expected = 100 * 0.000003 + 2_500 * 0.0000206 + 100 * 0.000195
    assert math.isclose(estimate.regulatory_fees, expected)


@pytest.mark.parametrize(
    ("premium", "quantity", "expected_commission"),
    [
        (0.03, 5, 1.25),
        (0.075, 2, 1.0),
        (2.0, 1, 1.0),
        (2.0, 2, 1.30),
    ],
)
def test_ibkr_us_option_premium_tiers_and_order_minimum(
    premium: float,
    quantity: int,
    expected_commission: float,
) -> None:
    registry = TransactionCostRegistry.load()
    estimate = registry.estimate_order_cost(
        registry.resolve("OPTIONS_US"),
        quantity=quantity,
        price=premium,
        option_premium=premium,
    )

    assert math.isclose(estimate.broker_commission, expected_commission)
    assert math.isclose(estimate.clearing_fees, quantity * 0.025)
    assert not estimate.complete
    assert "exchange and exchange-specific ORF fees" in estimate.omissions


def test_cn_futures_estimator_uses_instrument_fee_and_half_tick_per_side() -> None:
    registry = TransactionCostRegistry.load()
    instrument = InstrumentMaster("FUTURES_CN").get_profile("AP")
    estimate = registry.estimate_order_cost(
        registry.resolve("FUTURES_CN"),
        quantity=2,
        price=7_000.0,
        instrument_profile=instrument,
    )

    assert math.isclose(estimate.broker_commission, 10.0)
    assert math.isclose(estimate.slippage_cost, 10.0)
    assert math.isclose(estimate.total_cost, 20.0)
    assert estimate.complete


def test_attached_profile_is_fingerprinted_and_blocks_unsupported_net_claims(
    capsys: pytest.CaptureFixture[str],
) -> None:
    frame = pd.DataFrame({"ticker": ["AP"], "close": [7_000.0]})
    attached = attach_transaction_cost_policy(
        frame,
        market_vertical="FUTURES_CN",
        use_case="research_net",
    )

    assert attached.attrs["transaction_cost_profile_id"] == "cn_futures_broker_v1"
    assert len(attached.attrs["transaction_cost_profile_fingerprint"]) == 64
    assert attached.attrs["transaction_cost_production_ready"] is True

    with pytest.raises(TransactionCostReadinessError, match="not wired accurately"):
        attach_transaction_cost_policy(
            frame,
            market_vertical="EQUITY_US",
            use_case="research_net",
        )
    terminal = capsys.readouterr().err
    assert "[TRANSACTION COST READINESS BLOCK]" in terminal
    assert "Market: EQUITY_US" in terminal
    assert "What to add or finish:" in terminal
    assert "per-order minimums and caps" in terminal
    assert "transaction_cost_use_case='exploratory_gross'" in terminal
