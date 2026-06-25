"""Execution proposal models used before broker order placement."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from oqp.domain import (
    Instrument,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    utc_now,
)


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    BLOCKED = "blocked"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A draft order request produced by research or a dashboard."""

    instrument: Instrument
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"
    strategy_id: str | None = None
    signal_id: str | None = None
    target_weight: float | None = None
    reference_price: float | None = None
    confidence: float | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("OrderIntent.quantity must be positive")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("OrderIntent.confidence must be between 0 and 1")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("Limit order intents require a positive limit_price")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("Stop order intents require a positive stop_price")
        if self.reference_price is not None and self.reference_price <= 0:
            raise ValueError("OrderIntent.reference_price must be positive")

    @property
    def estimated_notional(self) -> float | None:
        price = self.reference_price or self.limit_price or self.stop_price
        if price is None:
            return None
        return self.quantity * price * self.instrument.multiplier

    def to_order(self, *, account_id: str | None = None, broker: str | None = None) -> Order:
        return Order(
            instrument=self.instrument,
            side=self.side,
            quantity=self.quantity,
            order_type=self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            status=OrderStatus.DRAFT,
            strategy_id=self.strategy_id,
            account_id=account_id,
            broker=broker,
            metadata={
                "signal_id": self.signal_id,
                "target_weight": self.target_weight,
                "confidence": self.confidence,
                "rationale": self.rationale,
                **self.metadata,
            },
        )


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """A locked bundle of draft order intents."""

    proposal_id: str
    source: str
    intents: tuple[OrderIntent, ...] = ()
    status: ProposalStatus = ProposalStatus.DRAFT
    paper_only: bool = True
    created_at: datetime = field(default_factory=utc_now)
    strategy_id: str | None = None
    research_run_id: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proposal_id:
            raise ValueError("TradeProposal.proposal_id is required")
        if not self.source:
            raise ValueError("TradeProposal.source is required")

    @property
    def estimated_notional(self) -> float | None:
        notionals = [intent.estimated_notional for intent in self.intents]
        if any(value is None for value in notionals):
            return None
        return sum(float(value) for value in notionals)
