"""Massive options data adapter stub."""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from typing import Any

from oqp.data.base import AdapterHealth, DataAdapterError, OptionsDataAdapter
from oqp.data.models import (
    OptionChainRequest,
    OptionContract,
    OptionQuote,
    OptionRight,
    Quote,
)
from oqp.domain import AssetClass, Instrument


@dataclass(frozen=True, slots=True)
class MassiveFlatFilesConfig:
    access_key_id: str | None = None
    secret_access_key: str | None = None
    endpoint: str = "https://files.massive.com"
    bucket: str = "flatfiles"

    @classmethod
    def from_env(cls) -> "MassiveFlatFilesConfig":
        return cls(
            access_key_id=os.getenv("MASSIVE_FLAT_FILES_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("MASSIVE_FLAT_FILES_SECRET_ACCESS_KEY"),
            endpoint=os.getenv(
                "MASSIVE_FLAT_FILES_ENDPOINT", "https://files.massive.com"
            ),
            bucket=os.getenv("MASSIVE_FLAT_FILES_BUCKET", "flatfiles"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.access_key_id and self.secret_access_key)


class MassiveOptionsDataAdapter(OptionsDataAdapter):
    """Thin Massive adapter shell for listed options data.

    The dashboard currently shows an Options Starter plan. We keep this adapter
    read-only and non-networked until credentials and endpoint coverage are
    explicitly wired through runtime config.
    """

    name = "massive_options"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.massive.com",
        flat_files: MassiveFlatFilesConfig | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("OPTIONS_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.flat_files = flat_files or MassiveFlatFilesConfig.from_env()

    def healthcheck(self) -> AdapterHealth:
        has_key = bool(self.api_key)
        return AdapterHealth(
            name=self.name,
            ok=has_key,
            message=(
                "Massive API key configured; REST and flat-file URL helpers are available."
                if has_key
                else "Missing MASSIVE_API_KEY or OPTIONS_API_KEY."
            ),
            metadata={
                "base_url": self.base_url,
                "flat_files_configured": self.flat_files.is_configured,
                "implemented": True,
                "abstract_methods_pending": ["get_option_chain"],
            },
        )

    def build_rest_url(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        if not self.api_key:
            raise DataAdapterError("Missing Massive API key.")

        clean_path = path.lstrip("/")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        return f"{self.base_url}/{clean_path}?{urllib.parse.urlencode(query)}"

    def _get_json_url(self, url: str, *, timeout: float = 15.0) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError as exc:
            raise DataAdapterError("Massive request failed.") from exc
        return payload if isinstance(payload, dict) else {}

    def _get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        return self._get_json_url(self.build_rest_url(path, params), timeout=timeout)

    def build_flat_file_url(self, path: str) -> str:
        clean_path = path.lstrip("/")
        return f"{self.flat_files.endpoint.rstrip('/')}/{self.flat_files.bucket}/{clean_path}"

    def get_option_expirations(
        self,
        underlying: str,
        *,
        limit: int = 1000,
        timeout: float = 15.0,
    ) -> list[str]:
        payload = self._get_json(
            "/v3/reference/options/contracts",
            {
                "underlying_ticker": underlying.upper(),
                "expired": "false",
                "sort": "expiration_date",
                "order": "asc",
                "limit": min(max(limit, 1), 1000),
            },
            timeout=timeout,
        )
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        expirations: list[str] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            expiration = row.get("expiration_date")
            if isinstance(expiration, str) and expiration not in expirations:
                expirations.append(expiration)
        return expirations

    def _snapshot_pages(
        self,
        underlying: str,
        params: dict[str, Any],
        *,
        max_pages: int = 100,
        timeout: float = 15.0,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        url = self.build_rest_url(f"/v3/snapshot/options/{underlying.upper()}", params)
        for _ in range(max_pages):
            payload = self._get_json_url(url, timeout=timeout)
            page_rows = payload.get("results")
            if isinstance(page_rows, list):
                rows.extend(row for row in page_rows if isinstance(row, dict))
            next_url = payload.get("next_url")
            if not next_url:
                return rows
            url = str(next_url)
            if "apiKey=" not in url and self.api_key:
                separator = "&" if "?" in url else "?"
                url = f"{url}{separator}{urllib.parse.urlencode({'apiKey': self.api_key})}"
        raise DataAdapterError(
            f"Massive option snapshot exceeded the {max_pages}-page safety cap"
        )

    def get_option_snapshot_rows(
        self,
        underlying: str,
        *,
        expiration: str | date | None = None,
        min_strike: float | None = None,
        max_strike: float | None = None,
        max_pages_per_type: int = 100,
        timeout: float = 15.0,
    ) -> list[dict[str, Any]]:
        """Return raw current-chain rows for immutable materialization."""

        params: dict[str, Any] = {
            "limit": 250,
            "sort": "strike_price",
            "order": "asc",
        }
        if expiration is not None:
            params["expiration_date"] = (
                expiration.isoformat()
                if isinstance(expiration, date)
                else str(expiration)
            )
        if min_strike is not None:
            params["strike_price.gte"] = float(min_strike)
        if max_strike is not None:
            params["strike_price.lte"] = float(max_strike)

        rows: list[dict[str, Any]] = []
        for contract_type in ("call", "put"):
            typed_params = dict(params)
            typed_params["contract_type"] = contract_type
            rows.extend(
                self._snapshot_pages(
                    underlying.upper(),
                    typed_params,
                    max_pages=max_pages_per_type,
                    timeout=timeout,
                )
            )
        return rows

    @staticmethod
    def _timestamp(value: Any) -> datetime:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)
        if parsed > 10_000_000_000_000:
            seconds = parsed / 1_000_000_000
        elif parsed > 10_000_000_000:
            seconds = parsed / 1_000
        else:
            seconds = parsed
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    @staticmethod
    def _num(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed

    def _quote_from_snapshot(self, underlying: Instrument, row: dict[str, Any]) -> OptionQuote | None:
        details = row.get("details")
        if not isinstance(details, dict):
            return None
        contract_type = str(details.get("contract_type") or "").lower()
        if contract_type not in {"call", "put"}:
            return None
        expiration_raw = details.get("expiration_date")
        try:
            expiration = date.fromisoformat(str(expiration_raw))
        except (TypeError, ValueError):
            return None
        strike = self._num(details.get("strike_price"))
        if strike is None or strike <= 0:
            return None

        quote_payload = row.get("last_quote")
        trade_payload = row.get("last_trade")
        day_payload = row.get("day")
        greeks = row.get("greeks")
        quote_payload = quote_payload if isinstance(quote_payload, dict) else {}
        trade_payload = trade_payload if isinstance(trade_payload, dict) else {}
        day_payload = day_payload if isinstance(day_payload, dict) else {}
        greeks = greeks if isinstance(greeks, dict) else {}

        bid = self._num(quote_payload.get("bid"))
        ask = self._num(quote_payload.get("ask"))
        last = self._num(trade_payload.get("price")) or self._num(day_payload.get("close"))
        mark = self._num(row.get("fmv"))
        if mark is None and bid is not None and ask is not None:
            mark = (bid + ask) / 2

        option_symbol = str(details.get("ticker") or "").strip()
        if not option_symbol:
            right_code = "C" if contract_type == "call" else "P"
            option_symbol = f"{underlying.symbol}{expiration:%y%m%d}{right_code}{int(strike * 1000):08d}"

        contract = OptionContract(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            right=OptionRight.CALL if contract_type == "call" else OptionRight.PUT,
            symbol=option_symbol,
            exchange=details.get("primary_exchange"),
            multiplier=float(details.get("shares_per_contract") or 100.0),
            currency="USD",
            metadata={"source": "massive"},
        )
        timestamp = self._timestamp(
            quote_payload.get("last_updated")
            or trade_payload.get("sip_timestamp")
            or day_payload.get("last_updated")
        )
        return OptionQuote(
            contract=contract,
            quote=Quote(
                instrument=Instrument(
                    symbol=option_symbol,
                    asset_class=AssetClass.OPTION,
                    currency=contract.currency,
                    multiplier=contract.multiplier,
                ),
                bid=bid,
                ask=ask,
                last=last,
                mark=mark,
                timestamp=timestamp,
                source=self.name,
                metadata={"raw": row},
            ),
            implied_volatility=self._num(row.get("implied_volatility")),
            delta=self._num(greeks.get("delta")),
            gamma=self._num(greeks.get("gamma")),
            theta=self._num(greeks.get("theta")),
            vega=self._num(greeks.get("vega")),
            open_interest=self._num(row.get("open_interest")),
            volume=self._num(day_payload.get("volume")),
            metadata={
                "break_even_price": row.get("break_even_price"),
                "source": "massive",
            },
        )

    def option_quotes_from_snapshot_rows(
        self,
        underlying: Instrument,
        rows: Sequence[dict[str, Any]],
    ) -> list[OptionQuote]:
        """Normalize previously fetched snapshot rows without another API call."""

        quotes = []
        for row in rows:
            quote = self._quote_from_snapshot(underlying, row)
            if quote is not None:
                quotes.append(quote)
        return quotes

    def get_option_chain(self, request: OptionChainRequest) -> Sequence[OptionQuote]:
        underlying = request.underlying.symbol.upper()
        rows = self.get_option_snapshot_rows(
            underlying,
            expiration=request.expiration,
            min_strike=request.min_strike,
            max_strike=request.max_strike,
        )
        return self.option_quotes_from_snapshot_rows(request.underlying, rows)
