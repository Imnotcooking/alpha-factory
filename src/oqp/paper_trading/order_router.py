"""Dry-run paper order ticket creation.

This module does not submit orders to IBKR. It converts a safety-approved
paper proposal into auditable order tickets and account events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from oqp.accounts import (
    AccountEnvironment,
    TradeEvent,
    write_account_trade_events,
)
from oqp.brokers import BrokerConnectionConfig
from oqp.domain import utc_now
from oqp.execution import TradeProposal
from oqp.paper_trading.execution_safety import PaperExecutionReview
from oqp.paper_trading.ledger import (
    load_paper_order_ticket,
    update_paper_order_ticket_status,
    write_paper_order_tickets,
)


class PaperOrderTicketStatus(str, Enum):
    DRY_RUN = "dry_run"
    BLOCKED = "blocked"
    APPROVED_FOR_SUBMIT = "approved_for_submit"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class PaperOrderTicket:
    order_id: str
    proposal_id: str
    review_id: str | None
    created_at: datetime
    strategy_id: str | None
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: float | None
    estimated_notional: float | None
    status: PaperOrderTicketStatus = PaperOrderTicketStatus.DRY_RUN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_ledger_row(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "created_at": self.created_at.isoformat(),
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "metadata": {
                "proposal_id": self.proposal_id,
                "review_id": self.review_id,
                "estimated_notional": self.estimated_notional,
                "broker_submit_enabled": False,
                **self.metadata,
            },
        }


@dataclass(frozen=True, slots=True)
class PaperOrderTicketResult:
    proposal_id: str
    review_id: str | None
    status: PaperOrderTicketStatus
    message: str
    order_ids: tuple[str, ...] = ()
    account_event_ids: tuple[str, ...] = ()
    paper_ledger_path: Path | None = None
    account_ledger_path: Path | None = None

    @property
    def order_count(self) -> int:
        return len(self.order_ids)

    @property
    def account_event_count(self) -> int:
        return len(self.account_event_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "review_id": self.review_id,
            "status": self.status.value,
            "message": self.message,
            "order_count": self.order_count,
            "order_ids": list(self.order_ids),
            "account_event_count": self.account_event_count,
            "account_event_ids": list(self.account_event_ids),
            "paper_ledger_path": None
            if self.paper_ledger_path is None
            else str(self.paper_ledger_path),
            "account_ledger_path": None
            if self.account_ledger_path is None
            else str(self.account_ledger_path),
        }


@dataclass(frozen=True, slots=True)
class PaperOrderTicketApprovalResult:
    order_id: str
    previous_status: str
    new_status: PaperOrderTicketStatus
    message: str
    account_event_id: str
    paper_ledger_path: Path
    account_ledger_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status.value,
            "message": self.message,
            "account_event_id": self.account_event_id,
            "paper_ledger_path": str(self.paper_ledger_path),
            "account_ledger_path": str(self.account_ledger_path),
        }


def create_dry_run_order_tickets(
    proposal: TradeProposal,
    *,
    review: PaperExecutionReview,
    paper_ledger_path: str | Path,
    account_ledger_path: str | Path,
    broker_config: BrokerConnectionConfig,
    account_id: str | None = None,
    review_id: str | None = None,
    created_at: datetime | None = None,
) -> PaperOrderTicketResult:
    timestamp = created_at or review.reviewed_at or utc_now()
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_readonly")

    if not review.passed:
        return PaperOrderTicketResult(
            proposal_id=proposal.proposal_id,
            review_id=review_id,
            status=PaperOrderTicketStatus.BLOCKED,
            message="Dry-run tickets were not created because the review is blocked.",
        )

    tickets = tuple(
        _ticket_from_intent(
            proposal,
            intent,
            index=index,
            review_id=review_id,
            created_at=timestamp,
        )
        for index, intent in enumerate(proposal.intents, start=1)
    )
    if not tickets:
        return PaperOrderTicketResult(
            proposal_id=proposal.proposal_id,
            review_id=review_id,
            status=PaperOrderTicketStatus.BLOCKED,
            message="Dry-run tickets were not created because the proposal has no intents.",
        )

    paper_write = write_paper_order_tickets(
        paper_ledger_path,
        [ticket.to_ledger_row() for ticket in tickets],
    )
    events = tuple(
        _event_from_ticket(
            ticket,
            account_id=account_id,
            broker=broker_config.broker,
            profile=profile,
            created_at=timestamp,
        )
        for ticket in tickets
    )
    account_write = write_account_trade_events(account_ledger_path, events)

    return PaperOrderTicketResult(
        proposal_id=proposal.proposal_id,
        review_id=review_id,
        status=PaperOrderTicketStatus.DRY_RUN,
        message="Dry-run order tickets created. No broker order was submitted.",
        order_ids=paper_write.order_ids,
        account_event_ids=account_write.event_ids,
        paper_ledger_path=paper_write.db_path,
        account_ledger_path=account_write.db_path,
    )


def set_paper_order_ticket_approval(
    *,
    order_id: str,
    status: PaperOrderTicketStatus | str,
    paper_ledger_path: str | Path,
    account_ledger_path: str | Path,
    broker_config: BrokerConnectionConfig,
    account_id: str | None = None,
    decided_by: str | None = None,
    reason: str | None = None,
    decided_at: datetime | None = None,
) -> PaperOrderTicketApprovalResult:
    new_status = _approval_status(status)
    timestamp = (decided_at or utc_now()).replace(microsecond=0)
    ticket_before = load_paper_order_ticket(paper_ledger_path, order_id)
    if ticket_before is None:
        raise ValueError(f"Paper order ticket not found: {order_id}")

    update_result = update_paper_order_ticket_status(
        paper_ledger_path,
        order_id,
        new_status.value,
        decided_at=timestamp,
        decided_by=decided_by,
        reason=reason,
    )
    ticket = load_paper_order_ticket(paper_ledger_path, order_id) or ticket_before
    metadata = dict(ticket.get("metadata") or {})
    profile = str(broker_config.metadata.get("profile") or "ibkr_paper_readonly")
    event = _approval_event_from_ticket(
        ticket,
        new_status=new_status,
        previous_status=update_result.previous_status,
        account_id=account_id,
        broker=broker_config.broker,
        profile=profile,
        decided_by=decided_by,
        reason=reason,
        decided_at=timestamp,
        metadata=metadata,
    )
    account_write = write_account_trade_events(account_ledger_path, (event,))
    message = (
        "Dry-run ticket approved for later paper submission. "
        "No broker order was submitted."
        if new_status == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT
        else "Dry-run ticket rejected. No broker order was submitted."
    )
    return PaperOrderTicketApprovalResult(
        order_id=str(order_id),
        previous_status=update_result.previous_status,
        new_status=new_status,
        message=message,
        account_event_id=account_write.event_ids[0],
        paper_ledger_path=update_result.db_path,
        account_ledger_path=account_write.db_path,
    )


def _ticket_from_intent(
    proposal: TradeProposal,
    intent: Any,
    *,
    index: int,
    review_id: str | None,
    created_at: datetime,
) -> PaperOrderTicket:
    instrument = intent.instrument
    strategy_id = intent.strategy_id or proposal.strategy_id
    price = intent.limit_price or intent.reference_price or intent.stop_price
    estimated_notional = intent.estimated_notional
    return PaperOrderTicket(
        order_id=f"paper-dryrun-{_safe_id(proposal.proposal_id)}-{index}",
        proposal_id=proposal.proposal_id,
        review_id=review_id,
        created_at=created_at,
        strategy_id=strategy_id,
        symbol=instrument.symbol,
        side=intent.side.value,
        quantity=float(intent.quantity),
        order_type=intent.order_type.value,
        limit_price=price,
        estimated_notional=estimated_notional,
        metadata={
            "source": "dry_run_order_ticket",
            "proposal_source": proposal.source,
            "research_run_id": proposal.research_run_id,
            "signal_id": intent.signal_id,
            "target_weight": intent.target_weight,
            "confidence": intent.confidence,
            "time_in_force": intent.time_in_force,
            "asset_class": instrument.asset_class.value,
            "currency": instrument.currency,
            "reference_price": intent.reference_price,
            "stop_price": intent.stop_price,
            "rationale": intent.rationale,
        },
    )


def _event_from_ticket(
    ticket: PaperOrderTicket,
    *,
    account_id: str | None,
    broker: str,
    profile: str,
    created_at: datetime,
) -> TradeEvent:
    return TradeEvent(
        event_id=f"evt-paper-order-ticket-{_safe_id(ticket.proposal_id)}-{_safe_id(ticket.order_id)}",
        event_type="paper_order_ticket",
        occurred_at=created_at,
        account_id=account_id,
        broker=broker,
        profile=profile,
        environment=AccountEnvironment.PAPER,
        symbol=ticket.symbol,
        side=ticket.side,
        quantity=ticket.quantity,
        price=ticket.limit_price,
        currency=str(ticket.metadata.get("currency") or "USD"),
        strategy_id=ticket.strategy_id,
        order_id=ticket.order_id,
        metadata={
            "proposal_id": ticket.proposal_id,
            "review_id": ticket.review_id,
            "status": ticket.status.value,
            "estimated_notional": ticket.estimated_notional,
            "broker_submit_enabled": False,
            "event_source": "dry_run_order_ticket",
            **ticket.metadata,
        },
    )


def _approval_event_from_ticket(
    ticket: dict[str, Any],
    *,
    new_status: PaperOrderTicketStatus,
    previous_status: str,
    account_id: str | None,
    broker: str,
    profile: str,
    decided_by: str | None,
    reason: str | None,
    decided_at: datetime,
    metadata: dict[str, Any],
) -> TradeEvent:
    event_type = (
        "paper_order_ticket_approved"
        if new_status == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT
        else "paper_order_ticket_rejected"
    )
    order_id = str(ticket["order_id"])
    return TradeEvent(
        event_id=(
            f"evt-{event_type}-{_safe_id(order_id)}-"
            f"{decided_at.strftime('%Y%m%dT%H%M%SZ')}"
        ),
        event_type=event_type,
        occurred_at=decided_at,
        account_id=account_id,
        broker=broker,
        profile=profile,
        environment=AccountEnvironment.PAPER,
        symbol=str(ticket["symbol"]),
        side=str(ticket.get("side") or ""),
        quantity=_optional_float(ticket.get("quantity")),
        price=_optional_float(ticket.get("limit_price")),
        currency=str(metadata.get("currency") or "USD"),
        strategy_id=ticket.get("strategy_id"),
        order_id=order_id,
        metadata={
            "proposal_id": metadata.get("proposal_id"),
            "review_id": metadata.get("review_id"),
            "previous_status": previous_status,
            "new_status": new_status.value,
            "decided_by": decided_by,
            "reason": reason,
            "broker_submit_enabled": False,
            "event_source": "paper_order_ticket_approval",
            "approval_only": True,
        },
    )


def _approval_status(value: PaperOrderTicketStatus | str) -> PaperOrderTicketStatus:
    try:
        status = (
            value
            if isinstance(value, PaperOrderTicketStatus)
            else PaperOrderTicketStatus(str(value))
        )
    except ValueError as exc:
        raise ValueError(
            "Paper order ticket approval status must be approved_for_submit or rejected."
        ) from exc
    if status not in {
        PaperOrderTicketStatus.APPROVED_FOR_SUBMIT,
        PaperOrderTicketStatus.REJECTED,
    }:
        raise ValueError(
            "Paper order ticket approval status must be approved_for_submit or rejected."
        )
    return status


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_id(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value)
    ).strip("-")
    return safe or "ticket"
