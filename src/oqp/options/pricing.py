"""Option mark fetching helpers.

Massive is the primary listed-options source for this project. Yahoo Finance is
kept as a best-effort fallback for simple marks when Massive does not return a
usable quote.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from oqp.data import MassiveOptionsDataAdapter, OptionChainRequest, OptionQuote
from oqp.domain import AssetClass, Instrument


def fetch_option_spread_mark(
    yf: Any,
    row: dict[str, Any],
    options_adapter: MassiveOptionsDataAdapter | None = None,
) -> float | None:
    """Fetch a net mark for a manual option spread row."""

    metadata = _json_dict(row.get("metadata_json"))
    if row.get("metadata"):
        metadata.update(row["metadata"])
    legs = metadata.get("legs")
    if not isinstance(legs, list):
        return None

    quote_symbol = str(row.get("quote_symbol") or row.get("underlying") or "").strip()
    expiry = str(row.get("expiry") or "").strip()
    net = 0.0
    priced = 0
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        price = fetch_option_mark(
            yf,
            quote_symbol,
            expiry,
            str(leg.get("option_type") or row.get("option_type") or "call").lower(),
            _float_or_none(leg.get("strike")),
            options_adapter=options_adapter,
            row_metadata=leg,
        )
        if price is None:
            continue
        quantity = _float_or_none(leg.get("quantity"))
        if quantity is None:
            quantity = 1.0 if str(leg.get("side") or "").lower() != "sell" else -1.0
        net += quantity * price
        priced += 1

    if priced != len(legs):
        return None

    row["metadata"] = metadata
    sources = {
        str(leg.get("pricing_method") or "").lower()
        for leg in legs
        if isinstance(leg, dict) and leg.get("pricing_method")
    }
    if sources == {"massive"}:
        row["pricing_method"] = "massive"
    elif "massive" in sources:
        row["pricing_method"] = "mixed_options"
    else:
        row["pricing_method"] = "yfinance"
    metadata["pricing_method"] = row["pricing_method"]
    return net


def fetch_option_mark(
    yf: Any,
    symbol: str,
    expiry: str,
    option_type: str,
    strike: float | None,
    *,
    options_adapter: MassiveOptionsDataAdapter | None = None,
    row_metadata: dict[str, Any] | None = None,
) -> float | None:
    """Fetch a single option mark, preferring Massive over Yahoo."""

    if not symbol or not expiry or strike is None:
        return None

    massive_quote = fetch_massive_option_quote(options_adapter, symbol, expiry, option_type, strike)
    if massive_quote is not None:
        price = option_quote_mark(massive_quote)
        if price is not None and price > 0:
            store_option_quote_metadata(row_metadata, massive_quote, price)
            return price

    price = fetch_yahoo_option_price(yf, symbol, expiry, option_type, strike)
    if price is not None and row_metadata is not None:
        row_metadata["current_price"] = price
        row_metadata["pricing_method"] = "yfinance"
    return price


def fetch_massive_option_quote(
    options_adapter: MassiveOptionsDataAdapter | None,
    symbol: str,
    expiry: str,
    option_type: str,
    strike: float,
) -> OptionQuote | None:
    if options_adapter is None:
        return None
    try:
        expiration = pd.to_datetime(expiry, errors="raise").date()
        request = OptionChainRequest(
            underlying=Instrument(symbol=symbol.upper(), asset_class=AssetClass.EQUITY),
            expiration=expiration,
            min_strike=strike,
            max_strike=strike,
        )
        quotes = options_adapter.get_option_chain(request)
    except Exception:
        return None

    wanted_type = "call" if option_type.startswith("c") else "put"
    candidates = []
    for quote in quotes:
        right = getattr(quote.contract.right, "value", str(quote.contract.right)).lower()
        if right != wanted_type:
            continue
        if abs(float(quote.contract.strike) - float(strike)) < 1e-8:
            candidates.append(quote)
    if not candidates:
        return None
    candidates.sort(key=lambda quote: option_quote_mark(quote) or 0.0, reverse=True)
    return candidates[0]


def option_quote_mark(quote: OptionQuote) -> float | None:
    for value in (quote.quote.mark, quote.quote.last):
        parsed = _float_or_none(value)
        if parsed is not None and parsed > 0:
            return parsed
    bid = _float_or_none(quote.quote.bid)
    ask = _float_or_none(quote.quote.ask)
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return None


def store_option_quote_metadata(
    row_metadata: dict[str, Any] | None,
    quote: OptionQuote,
    price: float,
) -> None:
    if row_metadata is None:
        return
    row_metadata.update(
        {
            "current_price": price,
            "pricing_method": "massive",
            "option_symbol": quote.contract.symbol,
            "bid": quote.quote.bid,
            "ask": quote.quote.ask,
            "last": quote.quote.last,
            "mark": quote.quote.mark,
            "implied_volatility": quote.implied_volatility,
            "iv": quote.implied_volatility,
            "delta": quote.delta,
            "gamma": quote.gamma,
            "theta": quote.theta,
            "vega": quote.vega,
            "open_interest": quote.open_interest,
            "volume": quote.volume,
            "quote_source": quote.quote.source or "massive_options",
            "quote_timestamp": quote.quote.timestamp.isoformat(),
        }
    )


def fetch_yahoo_option_price(
    yf: Any,
    symbol: str,
    expiry: str,
    option_type: str,
    strike: float | None,
) -> float | None:
    if not symbol or not expiry or strike is None:
        return None
    try:
        chain = yf.Ticker(symbol).option_chain(expiry)
    except Exception:
        return None
    table = chain.calls if option_type.startswith("c") else chain.puts
    if table is None or table.empty or "strike" not in table:
        return None
    strikes = pd.to_numeric(table["strike"], errors="coerce")
    match = table.loc[(strikes - strike).abs() < 1e-8]
    if match.empty:
        return None
    row = match.iloc[0]
    bid = _float_or_none(row.get("bid"))
    ask = _float_or_none(row.get("ask"))
    last = _float_or_none(row.get("lastPrice"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return last if last is not None and last > 0 else None


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed
