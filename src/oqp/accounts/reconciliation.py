"""Read-only reconciliation of canonical account snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from uuid import uuid4

from oqp.accounts.models import AccountSnapshot, PositionSnapshot
from oqp.accounts.reconciliation_models import (
    BreakCategory,
    BreakSeverity,
    NumericTolerance,
    ReconciliationBreak,
    ReconciliationPolicy,
    ReconciliationResult,
)


@dataclass(frozen=True, slots=True)
class _PositionTotal:
    quantity: float
    multiplier: float | None
    market_value: float | None


def reconcile_account_snapshots(
    reference: AccountSnapshot,
    observed: AccountSnapshot,
    *,
    policy: ReconciliationPolicy | None = None,
    reference_label: str = "source",
    observed_label: str = "ledger",
    run_id: str | None = None,
    compared_at: datetime | None = None,
) -> ReconciliationResult:
    """Compare two immutable account snapshots without mutating either side.

    The reference is the authoritative evidence for this run. The optional
    policy defaults to strict zero tolerance; callers must provide approved
    production tolerances explicitly.
    """

    active_policy = policy or ReconciliationPolicy()
    breaks: list[ReconciliationBreak] = []
    checks = 0

    checks += 1
    if reference.environment != observed.environment:
        breaks.append(
            _text_break(
                category=BreakCategory.IDENTITY,
                severity=BreakSeverity.CRITICAL,
                key="account",
                field="environment",
                reference_value=reference.environment.value,
                observed_value=observed.environment.value,
                reference_label=reference_label,
                observed_label=observed_label,
            )
        )

    checks += 1
    reference_currency = _normalize_text(reference.currency)
    observed_currency = _normalize_text(observed.currency)
    if reference_currency != observed_currency:
        breaks.append(
            _text_break(
                category=BreakCategory.IDENTITY,
                severity=BreakSeverity.CRITICAL,
                key="account",
                field="currency",
                reference_value=reference_currency,
                observed_value=observed_currency,
                reference_label=reference_label,
                observed_label=observed_label,
            )
        )

    max_time_delta = active_policy.max_snapshot_time_delta_seconds
    if max_time_delta is not None:
        checks += 1
        time_delta = _timestamp_delta_seconds(reference.as_of, observed.as_of)
        if time_delta > max_time_delta:
            breaks.append(
                ReconciliationBreak(
                    category=BreakCategory.IDENTITY,
                    severity=BreakSeverity.CRITICAL,
                    key="account",
                    field="snapshot_time_delta_seconds",
                    reference_value=reference.as_of.isoformat(),
                    observed_value=observed.as_of.isoformat(),
                    difference=time_delta,
                    tolerance=float(max_time_delta),
                    message=(
                        f"{reference_label} and {observed_label} snapshots are "
                        f"{time_delta:.3f} seconds apart."
                    ),
                )
            )

    position_checks, position_breaks = _reconcile_positions(
        reference.positions,
        observed.positions,
        policy=active_policy,
        reference_label=reference_label,
        observed_label=observed_label,
    )
    checks += position_checks
    breaks.extend(position_breaks)

    if active_policy.compare_cash:
        cash_checks, cash_breaks = _reconcile_cash(
            reference,
            observed,
            policy=active_policy,
            reference_label=reference_label,
            observed_label=observed_label,
        )
        checks += cash_checks
        breaks.extend(cash_breaks)

    if active_policy.compare_nav:
        nav_check, nav_break = _numeric_check(
            category=BreakCategory.NAV,
            severity=BreakSeverity.CRITICAL,
            key="account",
            field="net_liquidation",
            reference_value=reference.net_liquidation,
            observed_value=observed.net_liquidation,
            tolerance=active_policy.nav,
            reference_label=reference_label,
            observed_label=observed_label,
        )
        checks += nav_check
        if nav_break is not None:
            breaks.append(nav_break)

    timestamp = compared_at or datetime.now(timezone.utc).replace(microsecond=0)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return ReconciliationResult(
        run_id=run_id or f"recon_{uuid4().hex}",
        compared_at=timestamp,
        reference_snapshot_id=reference.snapshot_id,
        observed_snapshot_id=observed.snapshot_id,
        reference_label=reference_label,
        observed_label=observed_label,
        checks_performed=checks,
        breaks=tuple(breaks),
    )


def _reconcile_positions(
    reference_positions: tuple[PositionSnapshot, ...],
    observed_positions: tuple[PositionSnapshot, ...],
    *,
    policy: ReconciliationPolicy,
    reference_label: str,
    observed_label: str,
) -> tuple[int, list[ReconciliationBreak]]:
    reference_by_key = _aggregate_positions(reference_positions)
    observed_by_key = _aggregate_positions(observed_positions)
    breaks: list[ReconciliationBreak] = []
    checks = 0

    for key in sorted(set(reference_by_key) | set(observed_by_key)):
        checks += 1
        reference = reference_by_key.get(key)
        observed = observed_by_key.get(key)
        key_text = "/".join(key)
        if reference is None:
            if not policy.allow_additional_observed_positions:
                breaks.append(
                    ReconciliationBreak(
                        category=BreakCategory.POSITION,
                        severity=BreakSeverity.CRITICAL,
                        key=key_text,
                        field="presence",
                        reference_value=None,
                        observed_value="present",
                        difference=None,
                        tolerance=None,
                        message=(
                            f"Position {key_text} exists in {observed_label} but "
                            f"not in {reference_label}."
                        ),
                    )
                )
            continue
        if observed is None:
            breaks.append(
                ReconciliationBreak(
                    category=BreakCategory.POSITION,
                    severity=BreakSeverity.CRITICAL,
                    key=key_text,
                    field="presence",
                    reference_value="present",
                    observed_value=None,
                    difference=None,
                    tolerance=None,
                    message=(
                        f"Position {key_text} exists in {reference_label} but "
                        f"not in {observed_label}."
                    ),
                )
            )
            continue

        for field, reference_value, observed_value, tolerance, severity in (
            (
                "quantity",
                reference.quantity,
                observed.quantity,
                policy.quantity,
                BreakSeverity.CRITICAL,
            ),
            (
                "multiplier",
                reference.multiplier,
                observed.multiplier,
                policy.multiplier,
                BreakSeverity.CRITICAL,
            ),
        ):
            check, item = _numeric_check(
                category=BreakCategory.POSITION,
                severity=severity,
                key=key_text,
                field=field,
                reference_value=reference_value,
                observed_value=observed_value,
                tolerance=tolerance,
                reference_label=reference_label,
                observed_label=observed_label,
            )
            checks += check
            if item is not None:
                breaks.append(item)

        if policy.compare_market_value:
            check, item = _numeric_check(
                category=BreakCategory.POSITION,
                severity=BreakSeverity.WARNING,
                key=key_text,
                field="market_value",
                reference_value=reference.market_value,
                observed_value=observed.market_value,
                tolerance=policy.market_value,
                reference_label=reference_label,
                observed_label=observed_label,
            )
            checks += check
            if item is not None:
                breaks.append(item)

    return checks, breaks


def _reconcile_cash(
    reference: AccountSnapshot,
    observed: AccountSnapshot,
    *,
    policy: ReconciliationPolicy,
    reference_label: str,
    observed_label: str,
) -> tuple[int, list[ReconciliationBreak]]:
    reference_cash = _aggregate_cash(reference)
    observed_cash = _aggregate_cash(observed)
    breaks: list[ReconciliationBreak] = []
    checks = 0

    for currency in sorted(set(reference_cash) | set(observed_cash)):
        check, item = _numeric_check(
            category=BreakCategory.CASH,
            severity=BreakSeverity.CRITICAL,
            key=currency,
            field="cash_balance",
            reference_value=reference_cash.get(currency),
            observed_value=observed_cash.get(currency),
            tolerance=policy.cash,
            reference_label=reference_label,
            observed_label=observed_label,
        )
        checks += check
        if item is not None:
            breaks.append(item)

    check, item = _numeric_check(
        category=BreakCategory.CASH,
        severity=BreakSeverity.CRITICAL,
        key="account",
        field="cash_total",
        reference_value=reference.cash,
        observed_value=observed.cash,
        tolerance=policy.cash,
        reference_label=reference_label,
        observed_label=observed_label,
    )
    checks += check
    if item is not None:
        breaks.append(item)

    return checks, breaks


def _aggregate_positions(
    positions: tuple[PositionSnapshot, ...],
) -> dict[tuple[str, str, str], _PositionTotal]:
    grouped: dict[tuple[str, str, str], list[PositionSnapshot]] = {}
    for position in positions:
        key = (
            _normalize_text(position.symbol),
            _normalize_text(position.asset_class),
            _normalize_text(position.currency),
        )
        grouped.setdefault(key, []).append(position)

    totals: dict[tuple[str, str, str], _PositionTotal] = {}
    for key, rows in grouped.items():
        multipliers = {float(row.multiplier) for row in rows}
        values = [row.computed_market_value for row in rows]
        totals[key] = _PositionTotal(
            quantity=sum(float(row.quantity) for row in rows),
            multiplier=next(iter(multipliers)) if len(multipliers) == 1 else None,
            market_value=(
                sum(float(value) for value in values if value is not None)
                if all(value is not None for value in values)
                else None
            ),
        )
    return totals


def _aggregate_cash(snapshot: AccountSnapshot) -> dict[str, float]:
    balances: dict[str, float] = {}
    for item in snapshot.cash_balances:
        currency = _normalize_text(item.currency)
        balances[currency] = balances.get(currency, 0.0) + float(item.cash)
    return balances


def _numeric_check(
    *,
    category: BreakCategory,
    severity: BreakSeverity,
    key: str,
    field: str,
    reference_value: float | None,
    observed_value: float | None,
    tolerance: NumericTolerance,
    reference_label: str,
    observed_label: str,
) -> tuple[int, ReconciliationBreak | None]:
    if reference_value is None and observed_value is None:
        return 0, None

    if reference_value is None or observed_value is None:
        return 1, ReconciliationBreak(
            category=category,
            severity=severity,
            key=key,
            field=field,
            reference_value=reference_value,
            observed_value=observed_value,
            difference=None,
            tolerance=None,
            message=(
                f"{field} for {key} is unavailable on one side: "
                f"{reference_label}={reference_value!r}, "
                f"{observed_label}={observed_value!r}."
            ),
        )

    reference_number = float(reference_value)
    observed_number = float(observed_value)
    difference = abs(observed_number - reference_number)
    allowed = tolerance.threshold(reference_number)
    if tolerance.permits(reference_number, observed_number):
        return 1, None

    detail = "non-finite value" if not (
        isfinite(reference_number) and isfinite(observed_number)
    ) else f"difference {difference:.12g} exceeds tolerance {allowed:.12g}"
    return 1, ReconciliationBreak(
        category=category,
        severity=severity,
        key=key,
        field=field,
        reference_value=reference_number,
        observed_value=observed_number,
        difference=difference if isfinite(difference) else None,
        tolerance=allowed,
        message=(
            f"{field} for {key} does not reconcile: "
            f"{reference_label}={reference_number:.12g}, "
            f"{observed_label}={observed_number:.12g}; {detail}."
        ),
    )


def _text_break(
    *,
    category: BreakCategory,
    severity: BreakSeverity,
    key: str,
    field: str,
    reference_value: str,
    observed_value: str,
    reference_label: str,
    observed_label: str,
) -> ReconciliationBreak:
    return ReconciliationBreak(
        category=category,
        severity=severity,
        key=key,
        field=field,
        reference_value=reference_value,
        observed_value=observed_value,
        difference=None,
        tolerance=None,
        message=(
            f"{field} for {key} does not reconcile: "
            f"{reference_label}={reference_value!r}, "
            f"{observed_label}={observed_value!r}."
        ),
    )


def _normalize_text(value: str) -> str:
    return str(value).strip().upper()


def _timestamp_delta_seconds(reference: datetime, observed: datetime) -> float:
    reference_value = (
        reference.replace(tzinfo=timezone.utc)
        if reference.tzinfo is None
        else reference.astimezone(timezone.utc)
    )
    observed_value = (
        observed.replace(tzinfo=timezone.utc)
        if observed.tzinfo is None
        else observed.astimezone(timezone.utc)
    )
    return abs((reference_value - observed_value).total_seconds())
