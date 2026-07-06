"""Abstract interfaces for vendor data adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from oqp.data.models import (
    FeatureRequest,
    FundamentalSnapshot,
    FundamentalsRequest,
    OptionChainRequest,
    OptionQuote,
    PriceBar,
    PriceHistoryRequest,
    Quote,
)
from oqp.domain import Instrument
from oqp.domain.models import utc_now


class DataAdapterError(RuntimeError):
    """Base error for data adapter failures."""


@dataclass(frozen=True, slots=True)
class AdapterHealth:
    name: str
    ok: bool
    checked_at: datetime = field(default_factory=utc_now)
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataAdapter(ABC):
    """Small common surface shared by all data vendors."""

    name: str

    @abstractmethod
    def healthcheck(self) -> AdapterHealth:
        """Return whether the adapter can currently serve requests."""


class MarketDataAdapter(DataAdapter):
    """Adapter for historical bars and latest quotes."""

    @abstractmethod
    def get_price_history(self, request: PriceHistoryRequest) -> Sequence[PriceBar]:
        """Fetch historical bars for one instrument."""

    @abstractmethod
    def get_latest_quote(self, instrument: Instrument) -> Quote | None:
        """Fetch the most recent quote for one instrument."""


class FundamentalsAdapter(DataAdapter):
    """Adapter for fundamental and macro snapshots."""

    @abstractmethod
    def get_fundamentals(
        self, request: FundamentalsRequest
    ) -> FundamentalSnapshot | None:
        """Fetch a point-in-time fundamental snapshot."""


class OptionsDataAdapter(DataAdapter):
    """Adapter for listed options chains and quotes."""

    @abstractmethod
    def get_option_chain(self, request: OptionChainRequest) -> Sequence[OptionQuote]:
        """Fetch option quotes for an underlying instrument."""


class FeatureStoreAdapter(DataAdapter):
    """Adapter for reusable research and trading features."""

    @abstractmethod
    def get_features(self, request: FeatureRequest) -> Sequence[Mapping[str, Any]]:
        """Fetch feature rows as plain mappings."""
