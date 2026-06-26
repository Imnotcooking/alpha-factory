"""Paper order submission preflight checks.

This module deliberately does not place broker orders. It reviews approved
dry-run paper tickets and records why they are still blocked from submission.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from oqp.accounts import (
    AccountEnvironment,
    TradeEvent,
    write_account_trade_events,
)
from oqp.brokers import (
    BrokerAdapter,
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    BrokerEnvironment,
    BrokerHealth,
    BrokerAdapterError,
    OrderReceipt,
)
from oqp.config import OQPSettings
from oqp.domain import (
    AssetClass,
    Instrument,
    Order,
    OrderSide,
    OrderType,
    utc_now,
)
from oqp.paper_trading.ledger import (
    load_paper_order_ticket,
    update_paper_order_ticket_status,
)
from oqp.paper_trading.order_router import PaperOrderTicketStatus
from oqp.paper_trading.strategy_registry import PaperStrategyStatus


class PaperSubmissionDecision(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    SUBMITTED = "submitted"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PaperSubmissionCheck:
    name: str
    passed: bool
    detail: str
    severity: str = "block"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperSubmissionPreflight:
    order_id: str
    checked_at: datetime
    decision: PaperSubmissionDecision
    message: str
    checks: tuple[PaperSubmissionCheck, ...]
    symbol: str
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    strategy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision == PaperSubmissionDecision.READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "checked_at": self.checked_at.isoformat(),
            "decision": self.decision.value,
            "message": self.message,
            "checks": [check.to_dict() for check in self.checks],
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class PaperSubmissionPreflightRecordResult:
    db_path: Path
    event_id: str
    order_id: str
    decision: PaperSubmissionDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "event_id": self.event_id,
            "order_id": self.order_id,
            "decision": self.decision.value,
        }


@dataclass(frozen=True, slots=True)
class PaperOrderSubmissionResult:
    order_id: str
    submitted_at: datetime
    decision: PaperSubmissionDecision
    message: str
    preflight: PaperSubmissionPreflight
    receipt: OrderReceipt | None = None
    account_event_id: str | None = None
    paper_ledger_path: Path | None = None
    account_ledger_path: Path | None = None
    broker_health: BrokerHealth | None = None

    @property
    def submitted(self) -> bool:
        return self.decision == PaperSubmissionDecision.SUBMITTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "submitted_at": self.submitted_at.isoformat(),
            "decision": self.decision.value,
            "message": self.message,
            "preflight": self.preflight.to_dict(),
            "receipt": None if self.receipt is None else _receipt_dict(self.receipt),
            "account_event_id": self.account_event_id,
            "paper_ledger_path": None
            if self.paper_ledger_path is None
            else str(self.paper_ledger_path),
            "account_ledger_path": None
            if self.account_ledger_path is None
            else str(self.account_ledger_path),
            "broker_health": None
            if self.broker_health is None
            else {
                "status": self.broker_health.status.value,
                "account_id": self.broker_health.account_id,
                "message": self.broker_health.message,
            },
        }


def review_paper_order_submission(
    ticket: Mapping[str, Any],
    *,
    settings: OQPSettings,
    broker_config: BrokerConnectionConfig,
    strategy_record: Mapping[str, Any] | None = None,
    checked_at: datetime | None = None,
    broker_submit_implemented: bool = False,
) -> PaperSubmissionPreflight:
    timestamp = (checked_at or utc_now()).replace(microsecond=0)
    metadata = _metadata_dict(ticket.get("metadata") or ticket.get("metadata_json"))
    order_id = _text(ticket.get("order_id")) or "unknown-paper-ticket"
    status = _text(ticket.get("status")) or ""
    symbol = _text(ticket.get("symbol")) or "UNKNOWN"
    side = _text(ticket.get("side"))
    quantity = _optional_float(ticket.get("quantity"))
    price = _optional_float(ticket.get("limit_price"))
    strategy_id = _text(ticket.get("strategy_id"))
    strategy_status = _text((strategy_record or {}).get("status"))
    strategy_kill_switch = _bool((strategy_record or {}).get("kill_switch"))

    checks = (
        _check(
            "Ticket human-approved",
            status == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
            status or "missing",
        ),
        _check(
            "Strategy paper-running",
            (
                strategy_status == PaperStrategyStatus.RUNNING.value
                and not strategy_kill_switch
            ),
            _strategy_detail(strategy_record),
        ),
        _check(
            "Paper submit switch",
            settings.allow_paper_order_submit,
            f"ALLOW_PAPER_ORDER_SUBMIT={str(settings.allow_paper_order_submit).lower()}",
        ),
        _check(
            "Live trading disabled",
            not settings.allow_live_trading,
            f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
        ),
        _check(
            "Broker environment paper",
            broker_config.environment == BrokerEnvironment.PAPER,
            broker_config.environment.value,
        ),
        _check(
            "Broker profile write-enabled",
            not broker_config.readonly,
            f"readonly={str(broker_config.readonly).lower()}",
        ),
        _check(
            "Broker placement implementation",
            broker_submit_implemented,
            (
                "enabled for guarded paper submitter"
                if broker_submit_implemented
                else "disabled in current preflight; no IBKR order can be submitted here"
            ),
        ),
    )
    blockers = [
        check.name
        for check in checks
        if not check.passed and check.severity == "block"
    ]
    decision = (
        PaperSubmissionDecision.READY
        if not blockers
        else PaperSubmissionDecision.BLOCKED
    )
    message = (
        "Paper ticket is submit-ready, but no broker call is made by this preflight."
        if decision == PaperSubmissionDecision.READY
        else "Blocked by: " + ", ".join(blockers)
    )

    return PaperSubmissionPreflight(
        order_id=order_id,
        checked_at=timestamp,
        decision=decision,
        message=message,
        checks=checks,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        strategy_id=strategy_id,
        metadata={
            "proposal_id": metadata.get("proposal_id"),
            "review_id": metadata.get("review_id"),
            "approval_status": metadata.get("approval_status"),
            "strategy_registry_status": strategy_status,
            "strategy_kill_switch": strategy_kill_switch,
            "broker_submit_enabled": broker_submit_implemented,
            "submitter_skeleton": not broker_submit_implemented,
        },
    )


def submit_approved_paper_order_ticket(
    *,
    order_id: str,
    paper_ledger_path: str | Path,
    account_ledger_path: str | Path,
    settings: OQPSettings,
    broker_config: BrokerConnectionConfig,
    broker: BrokerAdapter,
    strategy_record: Mapping[str, Any] | None = None,
    account_id: str | None = None,
    submitted_at: datetime | None = None,
) -> PaperOrderSubmissionResult:
    """Submit one approved ticket to the IBKR paper profile after all gates pass."""

    timestamp = (submitted_at or utc_now()).replace(microsecond=0)
    ticket = load_paper_order_ticket(paper_ledger_path, order_id)
    if ticket is None:
        raise ValueError(f"Paper order ticket not found: {order_id}")

    preflight = review_paper_order_submission(
        ticket,
        settings=settings,
        broker_config=broker_config,
        strategy_record=strategy_record,
        checked_at=timestamp,
        broker_submit_implemented=True,
    )
    if not preflight.passed:
        return PaperOrderSubmissionResult(
            order_id=preflight.order_id,
            submitted_at=timestamp,
            decision=PaperSubmissionDecision.BLOCKED,
            message=preflight.message,
            preflight=preflight,
            paper_ledger_path=Path(paper_ledger_path),
            account_ledger_path=Path(account_ledger_path),
        )

    order = _order_from_ticket(ticket, account_id=account_id)
    health = broker.connect(broker_config)
    if health.status != BrokerConnectionStatus.CONNECTED:
        broker.disconnect()
        return PaperOrderSubmissionResult(
            order_id=preflight.order_id,
            submitted_at=timestamp,
            decision=PaperSubmissionDecision.ERROR,
            message=health.message or "Could not connect to IBKR paper profile.",
            preflight=preflight,
            paper_ledger_path=Path(paper_ledger_path),
            account_ledger_path=Path(account_ledger_path),
            broker_health=health,
        )

    try:
        receipt = broker.place_order(order)
    except BrokerAdapterError as exc:
        return PaperOrderSubmissionResult(
            order_id=preflight.order_id,
            submitted_at=timestamp,
            decision=PaperSubmissionDecision.ERROR,
            message=str(exc),
            preflight=preflight,
            paper_ledger_path=Path(paper_ledger_path),
            account_ledger_path=Path(account_ledger_path),
            broker_health=health,
        )
    finally:
        broker.disconnect()

    update_paper_order_ticket_status(
        paper_ledger_path,
        preflight.order_id,
        PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value,
        decided_at=timestamp,
        metadata_updates={
            "submitted_at": timestamp.isoformat(),
            "submitted_to_broker": True,
            "broker_order_id": receipt.broker_order_id,
            "broker_status": receipt.status.value,
            "broker_submit_enabled": True,
        },
    )
    event = _submission_event_from_receipt(
        preflight,
        receipt,
        broker_config=broker_config,
        account_id=account_id or health.account_id,
        submitted_at=timestamp,
    )
    account_write = write_account_trade_events(account_ledger_path, (event,))

    return PaperOrderSubmissionResult(
        order_id=preflight.order_id,
        submitted_at=timestamp,
        decision=PaperSubmissionDecision.SUBMITTED,
        message="Paper order submitted to IBKR paper profile.",
        preflight=preflight,
        receipt=receipt,
        account_event_id=account_write.event_ids[0],
        paper_ledger_path=Path(paper_ledger_path),
        account_ledger_path=account_write.db_path,
        broker_health=health,
    )


def record_paper_submission_preflight(
    account_ledger_path: str | Path,
    preflight: PaperSubmissionPreflight,
    *,
    broker_config: BrokerConnectionConfig,
    account_id: str | None = None,
) -> PaperSubmissionPreflightRecordResult:
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_readonly")
    event = TradeEvent(
        event_id=(
            f"evt-paper-submit-preflight-{preflight.decision.value}-"
            f"{_safe_id(preflight.order_id)}-"
            f"{preflight.checked_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        ),
        event_type=f"paper_order_submission_preflight_{preflight.decision.value}",
        occurred_at=preflight.checked_at,
        account_id=account_id,
        broker=broker_config.broker,
        profile=profile,
        environment=AccountEnvironment.PAPER,
        symbol=preflight.symbol,
        side=preflight.side,
        quantity=preflight.quantity,
        price=preflight.price,
        currency=str(preflight.metadata.get("currency") or "USD"),
        strategy_id=preflight.strategy_id,
        order_id=preflight.order_id,
        metadata={
            "decision": preflight.decision.value,
            "message": preflight.message,
            "checks": [check.to_dict() for check in preflight.checks],
            "broker_submit_enabled": False,
            "event_source": "paper_order_submission_preflight",
            "submitter_skeleton": True,
            **preflight.metadata,
        },
    )
    result = write_account_trade_events(account_ledger_path, (event,))
    return PaperSubmissionPreflightRecordResult(
        db_path=result.db_path,
        event_id=result.event_ids[0],
        order_id=preflight.order_id,
        decision=preflight.decision,
    )


def _check(name: str, passed: bool, detail: str) -> PaperSubmissionCheck:
    return PaperSubmissionCheck(name=name, passed=passed, detail=detail)


def _order_from_ticket(ticket: Mapping[str, Any], *, account_id: str | None) -> Order:
    metadata = _metadata_dict(ticket.get("metadata") or ticket.get("metadata_json"))
    asset_class = _asset_class(metadata.get("asset_class"))
    symbol = _text(ticket.get("symbol")) or "UNKNOWN"
    side = _order_side(ticket.get("side"))
    order_type = _order_type(ticket.get("order_type"))
    return Order(
        instrument=Instrument(
            symbol=symbol,
            asset_class=asset_class,
            currency=str(metadata.get("currency") or "USD"),
            broker_symbol=str(metadata.get("broker_symbol") or symbol),
            multiplier=_optional_float(metadata.get("multiplier")) or 1.0,
        ),
        side=side,
        quantity=_optional_float(ticket.get("quantity")) or 0.0,
        order_type=order_type,
        limit_price=_optional_float(ticket.get("limit_price")),
        time_in_force=str(metadata.get("time_in_force") or "DAY"),
        strategy_id=_text(ticket.get("strategy_id")),
        account_id=account_id,
        broker="ibkr",
        client_order_id=str(ticket.get("order_id") or ""),
        metadata={
            "proposal_id": metadata.get("proposal_id"),
            "review_id": metadata.get("review_id"),
            "paper_ticket_id": ticket.get("order_id"),
            "event_source": "paper_order_submitter",
        },
    )


def _submission_event_from_receipt(
    preflight: PaperSubmissionPreflight,
    receipt: OrderReceipt,
    *,
    broker_config: BrokerConnectionConfig,
    account_id: str | None,
    submitted_at: datetime,
) -> TradeEvent:
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_submit")
    return TradeEvent(
        event_id=(
            f"evt-paper-order-submitted-{_safe_id(preflight.order_id)}-"
            f"{submitted_at.strftime('%Y%m%dT%H%M%SZ')}"
        ),
        event_type="paper_order_submitted",
        occurred_at=submitted_at,
        account_id=account_id,
        broker=broker_config.broker,
        profile=profile,
        environment=AccountEnvironment.PAPER,
        symbol=preflight.symbol,
        side=preflight.side,
        quantity=preflight.quantity,
        price=preflight.price,
        currency=str(preflight.metadata.get("currency") or "USD"),
        strategy_id=preflight.strategy_id,
        order_id=preflight.order_id,
        broker_order_id=receipt.broker_order_id,
        metadata={
            "decision": PaperSubmissionDecision.SUBMITTED.value,
            "message": "Paper order submitted to IBKR paper profile.",
            "broker_submit_enabled": True,
            "event_source": "paper_order_submitter",
            "receipt": _receipt_dict(receipt),
            **preflight.metadata,
        },
    )


def _receipt_dict(receipt: OrderReceipt) -> dict[str, Any]:
    return {
        "status": receipt.status.value,
        "broker_order_id": receipt.broker_order_id,
        "client_order_id": receipt.client_order_id,
        "submitted_at": receipt.submitted_at.isoformat(),
        "message": receipt.message,
        "metadata": receipt.metadata,
    }


def _asset_class(value: Any) -> AssetClass:
    text = str(value or AssetClass.EQUITY.value).strip().lower()
    try:
        return AssetClass(text)
    except ValueError:
        return AssetClass.EQUITY


def _order_side(value: Any) -> OrderSide:
    text = str(value or "").strip().lower()
    return OrderSide.SELL if text == OrderSide.SELL.value else OrderSide.BUY


def _order_type(value: Any) -> OrderType:
    text = str(value or "").strip().lower()
    try:
        return OrderType(text)
    except ValueError:
        return OrderType.LIMIT


def _strategy_detail(strategy_record: Mapping[str, Any] | None) -> str:
    if not strategy_record:
        return "missing paper strategy registry record"
    status = _text(strategy_record.get("status")) or "missing"
    kill_switch = _bool(strategy_record.get("kill_switch"))
    return f"status={status}, kill_switch={str(kill_switch).lower()}"


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_id(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value)
    ).strip("-")
    return safe or "ticket"
