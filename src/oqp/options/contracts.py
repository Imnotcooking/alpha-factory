"""Typed contracts for listed-option research and backtesting.

The live options modules already expose pricing and book diagnostics. This
module defines the stable, vendor-neutral shapes that option chain loaders and
event-driven backtests can share.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from oqp.contracts.market_vertical import normalize_market_vertical
from oqp.data.models import OptionContract as DataOptionContract
from oqp.data.models import OptionQuote
from oqp.domain import AssetClass, Instrument


class OptionRight(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    AMERICAN = "american"
    EUROPEAN = "european"
    UNKNOWN = "unknown"


class SettlementStyle(str, Enum):
    PHYSICAL = "physical"
    CASH = "cash"
    UNKNOWN = "unknown"


OPTION_MARKET_VERTICALS = {"OPTIONS_US", "OPTIONS_CN"}


CHAIN_COLUMNS = (
    "date",
    "timestamp",
    "option_symbol",
    "vendor_symbol",
    "market_vertical",
    "exchange",
    "underlying_symbol",
    "underlying_type",
    "expiry",
    "right",
    "strike",
    "multiplier",
    "currency",
    "exercise_style",
    "settlement_style",
    "bid",
    "ask",
    "mid",
    "last",
    "mark",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "implied_volatility",
    "delta",
    "gamma",
    "theta",
    "vega",
    "quote_timestamp",
    "quote_source",
)


@dataclass(frozen=True, slots=True)
class OptionContractSpec:
    option_symbol: str
    underlying_symbol: str
    expiry: date
    right: OptionRight
    strike: float
    market_vertical: str
    multiplier: float = 100.0
    currency: str = "USD"
    exchange: str | None = None
    vendor_symbol: str | None = None
    underlying_type: str = "unknown"
    exercise_style: ExerciseStyle = ExerciseStyle.UNKNOWN
    settlement_style: SettlementStyle = SettlementStyle.UNKNOWN
    tick_size: float | None = None
    listing_date: date | None = None
    last_trade_date: date | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        market_vertical = normalize_market_vertical(self.market_vertical)
        if market_vertical not in OPTION_MARKET_VERTICALS:
            raise ValueError(
                f"OptionContractSpec.market_vertical must be one of {sorted(OPTION_MARKET_VERTICALS)}."
            )
        if self.strike <= 0:
            raise ValueError("OptionContractSpec.strike must be positive.")
        if self.multiplier <= 0:
            raise ValueError("OptionContractSpec.multiplier must be positive.")
        if not self.option_symbol:
            raise ValueError("OptionContractSpec.option_symbol is required.")
        if not self.underlying_symbol:
            raise ValueError("OptionContractSpec.underlying_symbol is required.")
        object.__setattr__(self, "market_vertical", market_vertical)
        object.__setattr__(self, "right", normalize_option_right(self.right))
        object.__setattr__(self, "underlying_symbol", self.underlying_symbol.upper())

    @property
    def key(self) -> str:
        return self.option_symbol

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat()
        payload["right"] = self.right.value
        payload["exercise_style"] = self.exercise_style.value
        payload["settlement_style"] = self.settlement_style.value
        if self.listing_date is not None:
            payload["listing_date"] = self.listing_date.isoformat()
        if self.last_trade_date is not None:
            payload["last_trade_date"] = self.last_trade_date.isoformat()
        return payload

    @classmethod
    def from_data_contract(
        cls,
        contract: DataOptionContract,
        *,
        market_vertical: str = "OPTIONS_US",
        exercise_style: str | ExerciseStyle = ExerciseStyle.AMERICAN,
        settlement_style: str | SettlementStyle = SettlementStyle.PHYSICAL,
    ) -> "OptionContractSpec":
        return cls(
            option_symbol=contract.symbol or _fallback_option_symbol(contract),
            vendor_symbol=contract.symbol,
            market_vertical=market_vertical,
            exchange=contract.exchange,
            underlying_symbol=contract.underlying.symbol,
            underlying_type=contract.underlying.asset_class.value,
            expiry=contract.expiration,
            right=normalize_option_right(contract.right.value),
            strike=float(contract.strike),
            multiplier=float(contract.multiplier),
            currency=contract.currency,
            exercise_style=normalize_exercise_style(exercise_style),
            settlement_style=normalize_settlement_style(settlement_style),
            metadata=dict(contract.metadata),
        )


@dataclass(frozen=True, slots=True)
class OptionQuoteSnapshot:
    contract: OptionContractSpec
    timestamp: datetime
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mark: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    implied_volatility: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    quote_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> date:
        return self.timestamp.date()

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return None

    @property
    def best_mark(self) -> float | None:
        for value in (self.mark, self.mid, self.close, self.last):
            if value is not None and value > 0:
                return float(value)
        return None

    def to_chain_row(self) -> dict[str, Any]:
        contract = self.contract
        return {
            "date": self.date,
            "timestamp": self.timestamp,
            "option_symbol": contract.option_symbol,
            "vendor_symbol": contract.vendor_symbol,
            "market_vertical": contract.market_vertical,
            "exchange": contract.exchange,
            "underlying_symbol": contract.underlying_symbol,
            "underlying_type": contract.underlying_type,
            "expiry": contract.expiry,
            "right": contract.right.value,
            "strike": contract.strike,
            "multiplier": contract.multiplier,
            "currency": contract.currency,
            "exercise_style": contract.exercise_style.value,
            "settlement_style": contract.settlement_style.value,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "mark": self.best_mark,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "quote_timestamp": self.timestamp,
            "quote_source": self.quote_source,
        }


def normalize_option_right(value: Any) -> OptionRight:
    if isinstance(value, OptionRight):
        return value
    normalized = str(value or "").strip().lower()
    aliases = {
        "c": "call",
        "calls": "call",
        "call_option": "call",
        "认购": "call",
        "p": "put",
        "puts": "put",
        "put_option": "put",
        "认沽": "put",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"call", "put"}:
        raise ValueError(f"Invalid option right: {value!r}.")
    return OptionRight(normalized)


def normalize_exercise_style(value: Any) -> ExerciseStyle:
    if isinstance(value, ExerciseStyle):
        return value
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"american", "us"}:
        return ExerciseStyle.AMERICAN
    if normalized in {"european", "eu"}:
        return ExerciseStyle.EUROPEAN
    return ExerciseStyle.UNKNOWN


def normalize_settlement_style(value: Any) -> SettlementStyle:
    if isinstance(value, SettlementStyle):
        return value
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"physical", "deliverable"}:
        return SettlementStyle.PHYSICAL
    if normalized in {"cash", "cash_settled"}:
        return SettlementStyle.CASH
    return SettlementStyle.UNKNOWN


def option_quote_to_snapshot(
    quote: OptionQuote,
    *,
    market_vertical: str = "OPTIONS_US",
) -> OptionQuoteSnapshot:
    contract = OptionContractSpec.from_data_contract(
        quote.contract,
        market_vertical=market_vertical,
    )
    return OptionQuoteSnapshot(
        contract=contract,
        timestamp=_ensure_aware_datetime(quote.quote.timestamp),
        bid=_positive_or_none(quote.quote.bid),
        ask=_positive_or_none(quote.quote.ask),
        last=_positive_or_none(quote.quote.last),
        mark=_positive_or_none(quote.quote.mark),
        volume=_positive_or_none(quote.volume),
        open_interest=_positive_or_none(quote.open_interest),
        implied_volatility=_positive_or_none(quote.implied_volatility),
        delta=_float_or_none(quote.delta),
        gamma=_float_or_none(quote.gamma),
        theta=_float_or_none(quote.theta),
        vega=_float_or_none(quote.vega),
        quote_source=quote.quote.source or str(quote.metadata.get("source") or ""),
        metadata=dict(quote.metadata),
    )


def contract_to_instrument(contract: OptionContractSpec) -> Instrument:
    return Instrument(
        symbol=contract.option_symbol,
        asset_class=AssetClass.OPTION,
        exchange=contract.exchange,
        currency=contract.currency,
        multiplier=contract.multiplier,
        metadata=contract.to_dict(),
    )


def _fallback_option_symbol(contract: DataOptionContract) -> str:
    right = "C" if normalize_option_right(contract.right.value) == OptionRight.CALL else "P"
    return f"{contract.underlying.symbol}{contract.expiration:%y%m%d}{right}{int(contract.strike * 1000):08d}"


def _ensure_aware_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _positive_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return parsed if parsed is not None and parsed > 0 else None
