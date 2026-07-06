"""Financial Modeling Prep adapter."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from typing import Any

from oqp.data.base import (
    AdapterHealth,
    DataAdapterError,
    FundamentalsAdapter,
    MarketDataAdapter,
)
from oqp.data.models import (
    FundamentalSnapshot,
    FundamentalsRequest,
    PriceBar,
    PriceHistoryRequest,
    Quote,
)
from oqp.domain import Instrument


class FMPDataAdapter(MarketDataAdapter, FundamentalsAdapter):
    """Thin FMP adapter for URL construction and JSON endpoints."""

    name = "fmp"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://financialmodelingprep.com/api",
        stable_base_url: str = "https://financialmodelingprep.com/stable",
    ) -> None:
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.stable_base_url = stable_base_url.rstrip("/")

    def healthcheck(self) -> AdapterHealth:
        has_key = bool(self.api_key)
        return AdapterHealth(
            name=self.name,
            ok=has_key,
            message=(
                "FMP API key configured; URL and JSON helpers are available."
                if has_key
                else "Missing FMP_API_KEY."
            ),
            metadata={
                "base_url": self.base_url,
                "stable_base_url": self.stable_base_url,
                "implemented": True,
                "available_helpers": [
                    "get_json",
                    "get_income_statement",
                    "get_cash_flow_statement",
                    "get_key_metrics",
                    "get_social_sentiment",
                    "get_news_sentiment",
                    "get_insider_trading",
                    "get_historical_rating",
                    "get_upgrades_downgrades",
                ],
            },
        )

    def build_url(
        self,
        endpoint: str,
        *,
        version: str = "v3",
        stable: bool = False,
        params: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise DataAdapterError("Missing FMP API key.")

        clean_endpoint = endpoint.lstrip("/")
        base_url = self.stable_base_url if stable else f"{self.base_url}/{version}"
        query = dict(params or {})
        query["apikey"] = self.api_key
        return f"{base_url}/{clean_endpoint}?{urllib.parse.urlencode(query)}"

    def get_json(
        self,
        endpoint: str,
        *,
        version: str = "v3",
        stable: bool = False,
        params: dict[str, Any] | None = None,
        timeout: float = 15.0,
    ) -> Any:
        url = self.build_url(endpoint, version=version, stable=stable, params=params)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                import json

                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise DataAdapterError(
                f"FMP request failed for endpoint: {endpoint} HTTP {exc.code}"
            ) from exc
        except OSError as exc:
            raise DataAdapterError(f"FMP request failed for endpoint: {endpoint}") from exc

    def get_stable_json(
        self,
        endpoint: str,
        *,
        symbol: str,
        limit: int | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 15.0,
    ) -> Any:
        query = dict(params or {})
        query["symbol"] = symbol.upper()
        if limit is not None:
            query["limit"] = limit

        return self.get_json(
            endpoint,
            stable=True,
            params=query,
            timeout=timeout,
        )

    def get_income_statement(self, symbol: str, *, limit: int = 5) -> Any:
        return self.get_stable_json("income-statement", symbol=symbol, limit=limit)

    def get_cash_flow_statement(self, symbol: str, *, limit: int = 5) -> Any:
        return self.get_stable_json("cash-flow-statement", symbol=symbol, limit=limit)

    def get_key_metrics(self, symbol: str, *, limit: int = 5) -> Any:
        return self.get_stable_json("key-metrics", symbol=symbol, limit=limit)

    def get_social_sentiment(self, symbol: str, *, page: int = 0) -> Any:
        return self.get_json(
            "historical/social-sentiment",
            version="v4",
            params={"symbol": symbol.upper(), "page": page},
        )

    def get_news_sentiment(self, symbol: str, *, page: int = 0) -> Any:
        return self.get_json(
            "historical/news-sentiment",
            version="v4",
            params={"symbol": symbol.upper(), "page": page},
        )

    def get_insider_trading(self, symbol: str, *, page: int = 0) -> Any:
        return self.get_json(
            "insider-trading",
            version="v4",
            params={"symbol": symbol.upper(), "page": page},
        )

    def get_historical_rating(self, symbol: str, *, limit: int = 1000) -> Any:
        return self.get_json(
            f"historical-rating/{symbol.upper()}",
            version="v3",
            params={"limit": limit},
        )

    def get_upgrades_downgrades(self, symbol: str) -> Any:
        return self.get_json(
            "upgrades-downgrades",
            version="v4",
            params={"symbol": symbol.upper()},
        )

    def get_price_history(self, request: PriceHistoryRequest) -> Sequence[PriceBar]:
        raise NotImplementedError("FMP price history integration is not wired yet.")

    def get_latest_quote(self, instrument: Instrument) -> Quote | None:
        raise NotImplementedError("FMP latest quote integration is not wired yet.")

    def get_fundamentals(
        self, request: FundamentalsRequest
    ) -> FundamentalSnapshot | None:
        raise NotImplementedError("FMP fundamentals integration is not wired yet.")
