"""Data-transfer objects for broker adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from oqp.domain import Order, OrderStatus, Position
from oqp.domain.models import utc_now


class BrokerEnvironment(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class BrokerConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class BrokerConnectionConfig:
    broker: str
    host: str
    port: int
    client_id: int
    environment: BrokerEnvironment = BrokerEnvironment.PAPER
    account_id: str | None = None
    readonly: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.broker:
            raise ValueError("BrokerConnectionConfig.broker is required")
        if not self.host:
            raise ValueError("BrokerConnectionConfig.host is required")
        if self.port <= 0:
            raise ValueError("BrokerConnectionConfig.port must be positive")


@dataclass(frozen=True, slots=True)
class BrokerHealth:
    broker: str
    status: BrokerConnectionStatus
    checked_at: datetime = field(default_factory=utc_now)
    account_id: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CashBalance:
    currency: str
    cash: float
    settled_cash: float | None = None
    buying_power: float | None = None
    as_of: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccountSummary:
    broker: str
    account_id: str
    currency: str
    net_liquidation: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    gross_position_value: float | None = None
    as_of: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    order: Order
    status: OrderStatus
    broker_order_id: str | None = None
    client_order_id: str | None = None
    submitted_at: datetime = field(default_factory=utc_now)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CancelResult:
    broker_order_id: str
    cancelled: bool
    status: OrderStatus | None = None
    cancelled_at: datetime = field(default_factory=utc_now)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    order: Order
    filled_quantity: float
    average_price: float
    broker_execution_id: str | None = None
    broker_order_id: str | None = None
    commission: float | None = None
    currency: str | None = None
    executed_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.filled_quantity <= 0:
            raise ValueError("ExecutionReport.filled_quantity must be positive")
        if self.average_price <= 0:
            raise ValueError("ExecutionReport.average_price must be positive")


@dataclass(frozen=True, slots=True)
class BrokerSnapshot:
    broker: str
    account: AccountSummary
    positions: tuple[Position, ...]
    cash_balances: tuple[CashBalance, ...] = ()
    open_orders: tuple[OrderReceipt, ...] = ()
    as_of: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
