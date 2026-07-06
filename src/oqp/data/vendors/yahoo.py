"""Yahoo Finance adapter stub."""

from __future__ import annotations

from collections.abc import Sequence

from oqp.data.base import AdapterHealth, MarketDataAdapter
from oqp.data.models import PriceBar, PriceHistoryRequest, Quote
from oqp.domain import Instrument


class YahooDataAdapter(MarketDataAdapter):
    """Thin Yahoo adapter shell for public market data."""

    name = "yahoo"

    def healthcheck(self) -> AdapterHealth:
        return AdapterHealth(
            name=self.name,
            ok=True,
            message="Yahoo adapter stub available; data methods are not wired yet.",
            metadata={"implemented": False},
        )

    def get_price_history(self, request: PriceHistoryRequest) -> Sequence[PriceBar]:
        raise NotImplementedError("Yahoo price history integration is not wired yet.")

    def get_latest_quote(self, instrument: Instrument) -> Quote | None:
        raise NotImplementedError("Yahoo latest quote integration is not wired yet.")
