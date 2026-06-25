"""Converters from existing broker/portfolio shapes into account snapshots."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.accounts.models import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    TradeEvent,
    account_timestamp,
)
from oqp.brokers import IBKRReadOnlyPortfolioSnapshot


def account_snapshot_from_ibkr_readonly(
    snapshot: IBKRReadOnlyPortfolioSnapshot,
    *,
    environment: AccountEnvironment | str,
    profile: str,
    broker: str = "ibkr",
    broker_label: str | None = None,
    snapshot_date: str | date | datetime | None = None,
    as_of: datetime | None = None,
) -> AccountSnapshot:
    """Convert the current read-only IBKR extract into the shared account shape."""

    if snapshot.error:
        raise ValueError(f"IBKR snapshot contains an error: {snapshot.error}")

    env = _environment(environment)
    timestamp = as_of or snapshot.health.checked_at or account_timestamp()
    metrics = snapshot.metrics
    cash = _optional_float(metrics.get("Available_Cash_USD"))
    nav = _optional_float(metrics.get("Total_NAV_USD"))
    buying_power = _optional_float(metrics.get("Buying_Power_USD"))
    margin_buffer = _optional_float(metrics.get("Margin_Buffer_USD"))
    positions = tuple(
        position_snapshot_from_legacy_row(row)
        for row in snapshot.position_rows
        if str(row.get("Ticker", "")).strip()
    )

    return AccountSnapshot(
        snapshot_id=_snapshot_id(
            environment=env,
            profile=profile,
            account_id=snapshot.health.account_id,
            timestamp=timestamp,
        ),
        as_of=timestamp,
        account_id=snapshot.health.account_id,
        broker=broker,
        profile=profile,
        environment=env,
        currency="USD",
        net_liquidation=nav,
        cash=cash,
        buying_power=buying_power,
        gross_position_value=_gross_value(positions),
        margin_buffer=margin_buffer,
        positions=positions,
        cash_balances=(
            (
                CashSnapshot(
                    currency="USD",
                    cash=cash,
                    buying_power=buying_power,
                ),
            )
            if cash is not None
            else ()
        ),
        metadata={
            "source": "ibkr_readonly",
            "broker_label": broker_label,
            "snapshot_date": _date_text(snapshot_date) if snapshot_date else None,
        },
    )


def account_snapshot_from_live_positions_frame(
    positions: pd.DataFrame,
    *,
    metrics: dict[str, Any] | None = None,
    environment: AccountEnvironment | str,
    profile: str,
    broker: str = "ibkr",
    broker_label: str | None = None,
    account_id: str | None = None,
    snapshot_date: str | date | datetime | None = None,
    as_of: datetime | None = None,
) -> AccountSnapshot:
    """Convert a legacy/live positions frame plus metrics into account shape."""

    env = _environment(environment)
    timestamp = as_of or account_timestamp()
    metrics = metrics or {}
    row_dicts = positions.to_dict("records") if not positions.empty else []
    position_snapshots = tuple(
        position_snapshot_from_legacy_row(row)
        for row in row_dicts
        if str(row.get("Ticker") or row.get("ticker") or "").strip()
    )
    cash = _optional_float(metrics.get("Available_Cash_USD"))
    nav = _optional_float(metrics.get("Total_NAV_USD"))
    buying_power = _optional_float(metrics.get("Buying_Power_USD"))
    margin_buffer = _optional_float(metrics.get("Margin_Buffer_USD"))

    return AccountSnapshot(
        snapshot_id=_snapshot_id(
            environment=env,
            profile=profile,
            account_id=account_id,
            timestamp=timestamp,
        ),
        as_of=timestamp,
        account_id=account_id,
        broker=broker,
        profile=profile,
        environment=env,
        currency="USD",
        net_liquidation=nav,
        cash=cash,
        buying_power=buying_power,
        gross_position_value=_gross_value(position_snapshots),
        margin_buffer=margin_buffer,
        positions=position_snapshots,
        cash_balances=(
            (
                CashSnapshot(
                    currency="USD",
                    cash=cash,
                    buying_power=buying_power,
                ),
            )
            if cash is not None
            else ()
        ),
        metadata={
            "source": "live_positions_frame",
            "broker_label": broker_label,
            "snapshot_date": _date_text(snapshot_date) if snapshot_date else None,
        },
    )


def account_trade_events_from_proposal_review(
    proposal: Any,
    *,
    decision: str,
    reviewed_at: datetime,
    environment: AccountEnvironment | str = AccountEnvironment.PAPER,
    profile: str = "ibkr_paper_readonly",
    broker: str = "ibkr",
    account_id: str | None = None,
    review_id: str | None = None,
    message: str | None = None,
) -> tuple[TradeEvent, ...]:
    """Represent a paper safety review as canonical non-fill account events."""

    env = _environment(environment)
    proposal_id = _text(getattr(proposal, "proposal_id", None), default="proposal")
    proposal_strategy_id = _optional_text(getattr(proposal, "strategy_id", None))
    research_run_id = _optional_text(getattr(proposal, "research_run_id", None))
    source = _optional_text(getattr(proposal, "source", None))
    events: list[TradeEvent] = []

    for index, intent in enumerate(tuple(getattr(proposal, "intents", ()) or ()), start=1):
        instrument = getattr(intent, "instrument", None)
        symbol = _text(getattr(instrument, "symbol", None), default="UNKNOWN")
        strategy_id = (
            _optional_text(getattr(intent, "strategy_id", None))
            or proposal_strategy_id
        )
        price = (
            _optional_float(getattr(intent, "reference_price", None))
            or _optional_float(getattr(intent, "limit_price", None))
            or _optional_float(getattr(intent, "stop_price", None))
        )
        metadata = {
            "proposal_id": proposal_id,
            "proposal_source": source,
            "decision": str(decision),
            "review_id": review_id,
            "message": message,
            "intent_index": index,
            "order_type": _enum_text(getattr(intent, "order_type", None)),
            "estimated_notional": _optional_float(getattr(intent, "estimated_notional", None)),
            "signal_id": _optional_text(getattr(intent, "signal_id", None)),
            "research_run_id": research_run_id,
            "target_weight": _optional_float(getattr(intent, "target_weight", None)),
            "confidence": _optional_float(getattr(intent, "confidence", None)),
            "paper_only": bool(getattr(proposal, "paper_only", True)),
        }
        events.append(
            TradeEvent(
                event_id=_event_id(
                    event_type="paper_review",
                    environment=env,
                    proposal_id=proposal_id,
                    index=index,
                    timestamp=reviewed_at,
                ),
                event_type="paper_review",
                occurred_at=reviewed_at,
                account_id=account_id,
                broker=broker,
                profile=profile,
                environment=env,
                symbol=symbol,
                side=_enum_text(getattr(intent, "side", None)),
                quantity=_optional_float(getattr(intent, "quantity", None)),
                price=price,
                currency=_optional_text(getattr(instrument, "currency", None)) or "USD",
                strategy_id=strategy_id,
                order_id=proposal_id,
                metadata=metadata,
            )
        )

    return tuple(events)


def position_snapshot_from_legacy_row(row: dict[str, Any]) -> PositionSnapshot:
    symbol = str(row.get("Ticker") or row.get("ticker") or "").strip()
    asset_class = _asset_class(row.get("AssetType") or row.get("asset_type") or "Equity")
    quantity = _float(row.get("Shares", row.get("shares")), default=0.0)
    average_cost = _optional_float(row.get("AvgPrice", row.get("avg_cost")))
    market_price = _optional_float(row.get("Broker_Price", row.get("current_price")))
    multiplier = _float(row.get("Multiplier", row.get("multiplier")), default=1.0) or 1.0
    market_value = _optional_float(row.get("market_value"))
    if market_value is None and market_price is not None:
        market_value = quantity * market_price * multiplier

    return PositionSnapshot(
        symbol=symbol,
        asset_class=asset_class,
        quantity=quantity,
        average_cost=average_cost,
        market_price=market_price,
        market_value=market_value,
        unrealized_pnl=_optional_float(row.get("Broker_PnL", row.get("unrealized_pnl"))),
        currency=str(row.get("Currency") or row.get("currency") or "USD").strip() or "USD",
        multiplier=multiplier,
        metadata={
            "broker_label": row.get("Broker") or row.get("broker"),
            "delta": _optional_float(row.get("delta")),
            "gamma": _optional_float(row.get("gamma")),
        },
    )


def _environment(value: AccountEnvironment | str) -> AccountEnvironment:
    if isinstance(value, AccountEnvironment):
        return value
    return AccountEnvironment(str(value).lower())


def _asset_class(value: Any) -> str:
    text = str(value or "equity").strip().lower()
    return {
        "stock": "equity",
        "stk": "equity",
        "equity": "equity",
        "etf": "etf",
        "option": "option",
        "opt": "option",
        "future": "future",
        "futures": "future",
        "fut": "future",
        "cash": "cash",
        "crypto": "crypto",
        "fx": "fx",
    }.get(text, text or "equity")


def _snapshot_id(
    *,
    environment: AccountEnvironment,
    profile: str,
    account_id: str | None,
    timestamp: datetime,
) -> str:
    account = (account_id or "unknown").replace(" ", "_")
    compact = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"acct-{environment.value}-{profile}-{account}-{compact}-{uuid4().hex[:8]}"


def _event_id(
    *,
    event_type: str,
    environment: AccountEnvironment,
    proposal_id: str,
    index: int,
    timestamp: datetime,
) -> str:
    compact = timestamp.strftime("%Y%m%dT%H%M%SZ")
    proposal = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in proposal_id
    ).strip("-")
    return (
        f"evt-{environment.value}-{event_type}-{proposal or 'proposal'}-"
        f"{index}-{compact}-{uuid4().hex[:8]}"
    )


def _gross_value(positions: tuple[PositionSnapshot, ...]) -> float | None:
    values = [
        abs(value)
        for position in positions
        if (value := position.computed_market_value) is not None
    ]
    return sum(values) if values else None


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text(value: Any, *, default: str) -> str:
    return _optional_text(value) or default


def _enum_text(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    return _optional_text(enum_value if enum_value is not None else value)


def _float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed
