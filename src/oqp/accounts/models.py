"""Canonical account snapshot contracts shared by live and paper lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from oqp.domain.models import utc_now


class AccountEnvironment(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    SIM = "sim"


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    symbol: str
    asset_class: str
    quantity: float
    average_cost: float | None = None
    market_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    currency: str = "USD"
    multiplier: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("PositionSnapshot.symbol is required")
        if not self.asset_class.strip():
            raise ValueError("PositionSnapshot.asset_class is required")
        if self.multiplier <= 0:
            raise ValueError("PositionSnapshot.multiplier must be positive")

    @property
    def computed_market_value(self) -> float | None:
        if self.market_value is not None:
            return float(self.market_value)
        if self.market_price is None:
            return None
        return float(self.quantity) * float(self.market_price) * float(self.multiplier)


@dataclass(frozen=True, slots=True)
class CashSnapshot:
    currency: str
    cash: float
    settled_cash: float | None = None
    buying_power: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("CashSnapshot.currency is required")


@dataclass(frozen=True, slots=True)
class NavSnapshot:
    date: str
    account_key: str
    account_id: str | None
    broker: str
    profile: str
    environment: AccountEnvironment
    as_of: datetime
    net_liquidation: float
    cash: float | None
    daily_pnl: float | None
    position_count: int
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class TradeEvent:
    event_id: str
    event_type: str
    occurred_at: datetime
    account_id: str | None
    broker: str
    profile: str
    environment: AccountEnvironment
    symbol: str
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    commission: float | None = None
    currency: str | None = None
    strategy_id: str | None = None
    order_id: str | None = None
    broker_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("TradeEvent.event_id is required")
        if not self.event_type.strip():
            raise ValueError("TradeEvent.event_type is required")
        if not self.broker.strip():
            raise ValueError("TradeEvent.broker is required")
        if not self.profile.strip():
            raise ValueError("TradeEvent.profile is required")
        if not self.symbol.strip():
            raise ValueError("TradeEvent.symbol is required")

    @property
    def account_key(self) -> str:
        account = self.account_id or "unknown"
        return f"{self.environment.value}:{self.broker}:{self.profile}:{account}"


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    snapshot_id: str
    as_of: datetime
    account_id: str | None
    broker: str
    profile: str
    environment: AccountEnvironment
    currency: str = "USD"
    net_liquidation: float | None = None
    cash: float | None = None
    buying_power: float | None = None
    gross_position_value: float | None = None
    margin_buffer: float | None = None
    positions: tuple[PositionSnapshot, ...] = ()
    cash_balances: tuple[CashSnapshot, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("AccountSnapshot.snapshot_id is required")
        if not self.broker.strip():
            raise ValueError("AccountSnapshot.broker is required")
        if not self.profile.strip():
            raise ValueError("AccountSnapshot.profile is required")
        if not self.currency.strip():
            raise ValueError("AccountSnapshot.currency is required")

    @property
    def account_key(self) -> str:
        account = self.account_id or "unknown"
        return f"{self.environment.value}:{self.broker}:{self.profile}:{account}"

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def snapshot_date(self) -> str:
        return self.as_of.date().isoformat()

    @property
    def computed_gross_position_value(self) -> float | None:
        if self.gross_position_value is not None:
            return float(self.gross_position_value)
        values = [
            abs(value)
            for position in self.positions
            if (value := position.computed_market_value) is not None
        ]
        if not values:
            return None
        return sum(values)

    def nav_snapshot(
        self,
        *,
        snapshot_date: str | date | datetime | None = None,
        daily_pnl: float | None = None,
    ) -> NavSnapshot:
        date_value = _date_text(snapshot_date or self.as_of)
        return NavSnapshot(
            date=date_value,
            account_key=self.account_key,
            account_id=self.account_id,
            broker=self.broker,
            profile=self.profile,
            environment=self.environment,
            as_of=self.as_of,
            net_liquidation=float(self.net_liquidation or 0.0),
            cash=None if self.cash is None else float(self.cash),
            daily_pnl=daily_pnl,
            position_count=self.position_count,
            snapshot_id=self.snapshot_id,
        )


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def account_timestamp() -> datetime:
    return utc_now().replace(microsecond=0)
