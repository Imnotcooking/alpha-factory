"""Portfolio snapshot contracts for investing and middle-office views."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd


LEGACY_POSITION_COLUMNS = [
    "Ticker",
    "Shares",
    "AvgPrice",
    "Broker_Price",
    "Broker_PnL",
    "Currency",
    "AssetType",
    "Multiplier",
    "delta",
    "gamma",
    "Broker",
]

LIVE_POSITION_COLUMNS = [
    "date",
    "broker",
    "ticker",
    "asset_type",
    "shares",
    "avg_cost",
    "current_price",
    "unrealized_pnl",
    "currency",
    "delta",
    "gamma",
]


@dataclass(frozen=True, slots=True)
class PortfolioPositionSnapshot:
    broker: str
    ticker: str
    shares: float
    avg_price: float
    currency: str = "USD"
    asset_type: str = "Equity"
    multiplier: float = 1.0
    broker_price: float = 0.0
    broker_pnl: float = 0.0
    delta: float = 1.0
    gamma: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.broker.strip():
            raise ValueError("PortfolioPositionSnapshot.broker is required")
        if not self.ticker.strip():
            raise ValueError("PortfolioPositionSnapshot.ticker is required")
        if self.multiplier <= 0:
            raise ValueError("PortfolioPositionSnapshot.multiplier must be positive")

    @property
    def market_value(self) -> float:
        return self.shares * self.broker_price * self.multiplier

    def to_legacy_row(self) -> dict[str, Any]:
        return {
            "Ticker": self.ticker,
            "Shares": self.shares,
            "AvgPrice": self.avg_price,
            "Broker_Price": self.broker_price,
            "Broker_PnL": self.broker_pnl,
            "Currency": self.currency,
            "AssetType": self.asset_type,
            "Multiplier": self.multiplier,
            "delta": self.delta,
            "gamma": self.gamma,
            "Broker": self.broker,
        }

    def to_live_position_row(self, snapshot_date: str | date | datetime) -> dict[str, Any]:
        if isinstance(snapshot_date, datetime):
            date_value = snapshot_date.date().isoformat()
        elif isinstance(snapshot_date, date):
            date_value = snapshot_date.isoformat()
        else:
            date_value = str(snapshot_date)

        return {
            "date": date_value,
            "broker": self.broker,
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "shares": self.shares,
            "avg_cost": self.avg_price,
            "current_price": self.broker_price,
            "unrealized_pnl": self.broker_pnl,
            "currency": self.currency,
            "delta": self.delta,
            "gamma": self.gamma,
        }


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    as_of: date
    positions: tuple[PortfolioPositionSnapshot, ...] = ()
    broker_metrics: dict[str, Any] = field(default_factory=dict)
    banked_profits: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gross_market_value(self) -> float:
        return sum(abs(position.market_value) for position in self.positions)


def position_snapshots_to_legacy_frame(
    positions: list[PortfolioPositionSnapshot] | tuple[PortfolioPositionSnapshot, ...],
) -> pd.DataFrame:
    rows = [position.to_legacy_row() for position in positions]
    return pd.DataFrame(rows, columns=LEGACY_POSITION_COLUMNS)


def position_snapshots_to_live_positions_frame(
    positions: list[PortfolioPositionSnapshot] | tuple[PortfolioPositionSnapshot, ...],
    snapshot_date: str | date | datetime,
) -> pd.DataFrame:
    rows = [position.to_live_position_row(snapshot_date) for position in positions]
    return pd.DataFrame(rows, columns=LIVE_POSITION_COLUMNS)
