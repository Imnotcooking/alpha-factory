from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from oqp.accounts.models import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
)
from oqp.accounts.reconciliation import reconcile_account_snapshots
from oqp.accounts.reconciliation_models import (
    BreakCategory,
    NumericTolerance,
    ReconciliationPolicy,
    ReconciliationStatus,
)


AS_OF = datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc)


def test_identical_account_snapshots_reconcile() -> None:
    reference = _snapshot("source")
    observed = replace(reference, snapshot_id="ledger")

    result = reconcile_account_snapshots(reference, observed, run_id="recon_test")

    assert result.status is ReconciliationStatus.PASS
    assert result.break_count == 0
    assert result.checks_performed > 0
    assert result.run_id == "recon_test"


def test_quantity_cash_and_nav_differences_are_reported() -> None:
    reference = _snapshot("source")
    observed = _snapshot(
        "ledger",
        quantity=9.0,
        cash=900.0,
        nav=1900.0,
    )

    result = reconcile_account_snapshots(reference, observed)
    break_fields = {(item.category, item.field) for item in result.breaks}

    assert result.status is ReconciliationStatus.BREAK
    assert (BreakCategory.POSITION, "quantity") in break_fields
    assert (BreakCategory.CASH, "cash_balance") in break_fields
    assert (BreakCategory.CASH, "cash_total") in break_fields
    assert (BreakCategory.NAV, "net_liquidation") in break_fields
    assert result.critical_break_count >= 4


def test_explicit_relative_tolerance_allows_small_valuation_difference() -> None:
    reference = _snapshot("source", market_value=1000.0, nav=2000.0)
    observed = _snapshot("ledger", market_value=1005.0, nav=2005.0)
    policy = ReconciliationPolicy(
        market_value=NumericTolerance(relative=0.01),
        nav=NumericTolerance(relative=0.01),
    )

    result = reconcile_account_snapshots(reference, observed, policy=policy)

    assert result.status is ReconciliationStatus.PASS


def test_aggregate_policy_can_allow_approved_additional_positions() -> None:
    reference = _snapshot("source")
    manual = PositionSnapshot(
        symbol="EXTERNAL",
        asset_class="EQUITY",
        quantity=1.0,
        market_value=50.0,
        currency="USD",
        metadata={"manual_external": True},
    )
    observed = replace(
        reference,
        snapshot_id="unified",
        profile="unified_live",
        positions=(*reference.positions, manual),
    )

    strict_result = reconcile_account_snapshots(reference, observed)
    aggregate_result = reconcile_account_snapshots(
        reference,
        observed,
        policy=ReconciliationPolicy(allow_additional_observed_positions=True),
    )

    assert any(item.field == "presence" for item in strict_result.breaks)
    assert aggregate_result.status is ReconciliationStatus.PASS


def test_snapshot_time_cut_is_checked_only_when_configured() -> None:
    reference = _snapshot("source")
    observed = replace(
        reference,
        snapshot_id="ledger",
        as_of=datetime(2026, 7, 17, 8, 5, tzinfo=timezone.utc),
    )

    result = reconcile_account_snapshots(
        reference,
        observed,
        policy=ReconciliationPolicy(max_snapshot_time_delta_seconds=60),
    )

    assert any(item.field == "snapshot_time_delta_seconds" for item in result.breaks)


def test_snapshot_time_cut_accepts_mixed_naive_and_aware_inputs() -> None:
    reference = _snapshot("source")
    observed = replace(
        reference,
        snapshot_id="ledger",
        as_of=datetime(2026, 7, 17, 8, 0),
    )

    result = reconcile_account_snapshots(
        reference,
        observed,
        policy=ReconciliationPolicy(max_snapshot_time_delta_seconds=0),
    )

    assert result.status is ReconciliationStatus.PASS


def _snapshot(
    snapshot_id: str,
    *,
    quantity: float = 10.0,
    market_value: float = 1000.0,
    cash: float = 1000.0,
    nav: float = 2000.0,
) -> AccountSnapshot:
    return AccountSnapshot(
        snapshot_id=snapshot_id,
        as_of=AS_OF,
        account_id="account-1",
        broker="ibkr",
        profile="ibkr_live_readonly",
        environment=AccountEnvironment.LIVE,
        currency="USD",
        net_liquidation=nav,
        cash=cash,
        positions=(
            PositionSnapshot(
                symbol="AAPL",
                asset_class="EQUITY",
                quantity=quantity,
                market_value=market_value,
                currency="USD",
            ),
        ),
        cash_balances=(CashSnapshot(currency="USD", cash=cash),),
    )
