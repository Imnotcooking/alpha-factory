"""Data-transfer objects for market data and feature adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from oqp.domain import Instrument
from oqp.domain.models import utc_now


class BarInterval(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    ONE_HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class PriceHistoryRequest:
    instrument: Instrument
    start: date | datetime
    end: date | datetime | None = None
    interval: BarInterval = BarInterval.DAILY
    adjusted: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PriceBar:
    instrument: Instrument
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    adjusted_close: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("PriceBar.high cannot be below PriceBar.low")


@dataclass(frozen=True, slots=True)
class Quote:
    instrument: Instrument
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mark: float | None = None
    timestamp: datetime = field(default_factory=utc_now)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FundamentalsRequest:
    instrument: Instrument
    as_of: date | datetime | None = None
    fields: Sequence[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FundamentalSnapshot:
    instrument: Instrument
    as_of: date | datetime
    values: Mapping[str, Any]
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OptionContract:
    underlying: Instrument
    expiration: date
    strike: float
    right: OptionRight
    symbol: str | None = None
    exchange: str | None = None
    multiplier: float = 100.0
    currency: str = "USD"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise ValueError("OptionContract.strike must be positive")
        if self.multiplier <= 0:
            raise ValueError("OptionContract.multiplier must be positive")


@dataclass(frozen=True, slots=True)
class OptionChainRequest:
    underlying: Instrument
    expiration: date | None = None
    as_of: date | datetime | None = None
    min_strike: float | None = None
    max_strike: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OptionQuote:
    contract: OptionContract
    quote: Quote
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    open_interest: float | None = None
    volume: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FeatureRequest:
    feature_set: str
    instruments: Sequence[Instrument]
    start: date | datetime
    end: date | datetime | None = None
    frequency: BarInterval = BarInterval.DAILY
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature_set:
            raise ValueError("FeatureRequest.feature_set is required")
        if not self.instruments:
            raise ValueError("FeatureRequest.instruments cannot be empty")
