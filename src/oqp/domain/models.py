"""Core domain models shared by dashboards, research, and execution.

These dataclasses intentionally avoid framework dependencies so old and new
parts of the repo can adopt them gradually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    FX = "fx"
    CRYPTO = "crypto"
    CASH = "cash"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    exchange: str | None = None
    currency: str = "USD"
    broker_symbol: str | None = None
    multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Instrument.symbol is required")
        if self.multiplier <= 0:
            raise ValueError("Instrument.multiplier must be positive")


@dataclass(frozen=True, slots=True)
class Position:
    instrument: Instrument
    quantity: float
    average_cost: float
    market_price: float | None = None
    account_id: str | None = None
    broker: str | None = None
    as_of: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def market_value(self) -> float | None:
        if self.market_price is None:
            return None
        return self.quantity * self.market_price * self.instrument.multiplier

    @property
    def unrealized_pnl(self) -> float | None:
        if self.market_price is None:
            return None
        return (
            self.quantity
            * (self.market_price - self.average_cost)
            * self.instrument.multiplier
        )


@dataclass(frozen=True, slots=True)
class Order:
    instrument: Instrument
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.DRAFT
    strategy_id: str | None = None
    account_id: str | None = None
    broker: str | None = None
    client_order_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Order.quantity must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if self.limit_price is None or self.limit_price <= 0:
                raise ValueError("Limit orders require a positive limit_price")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if self.stop_price is None or self.stop_price <= 0:
                raise ValueError("Stop orders require a positive stop_price")


@dataclass(frozen=True, slots=True)
class Signal:
    instrument: Instrument
    direction: int
    strength: float
    strategy_id: str
    generated_at: datetime = field(default_factory=utc_now)
    target_weight: float | None = None
    horizon: str | None = None
    research_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {-1, 0, 1}:
            raise ValueError("Signal.direction must be -1, 0, or 1")
        if not 0 <= self.strength <= 1:
            raise ValueError("Signal.strength must be between 0 and 1")
        if not self.strategy_id:
            raise ValueError("Signal.strategy_id is required")


@dataclass(frozen=True, slots=True)
class StrategyArtifact:
    artifact_id: str
    strategy_id: str
    path: str
    created_at: datetime = field(default_factory=utc_now)
    version: str | None = None
    research_run_id: str | None = None
    status: str = "candidate"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("StrategyArtifact.artifact_id is required")
        if not self.strategy_id:
            raise ValueError("StrategyArtifact.strategy_id is required")
        if not self.path:
            raise ValueError("StrategyArtifact.path is required")
