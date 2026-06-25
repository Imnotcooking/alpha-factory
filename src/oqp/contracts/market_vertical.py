"""Market vertical names shared by research, trading, and dashboards."""

from __future__ import annotations

from enum import Enum
from typing import Any


class MarketVertical(str, Enum):
    EQUITY_US = "EQUITY_US"
    EQUITY_CN = "EQUITY_CN"
    EQUITY_HK = "EQUITY_HK"
    FUTURES_CN = "FUTURES_CN"
    FUTURES_US = "FUTURES_US"
    OPTIONS_US = "OPTIONS_US"
    OPTIONS_CN = "OPTIONS_CN"
    FX_SPOT = "FX_SPOT"
    CRYPTO_PERP = "CRYPTO_PERP"
    MULTI_ASSET = "MULTI_ASSET"
    UNKNOWN = "UNKNOWN"


_ALIASES = {
    "US_EQUITY": MarketVertical.EQUITY_US.value,
    "US_EQUITIES": MarketVertical.EQUITY_US.value,
    "STOCKS_US": MarketVertical.EQUITY_US.value,
    "US_STOCKS": MarketVertical.EQUITY_US.value,
    "CN_EQUITY": MarketVertical.EQUITY_CN.value,
    "CN_EQUITIES": MarketVertical.EQUITY_CN.value,
    "CHINA_EQUITY": MarketVertical.EQUITY_CN.value,
    "CHINA_EQUITIES": MarketVertical.EQUITY_CN.value,
    "HK_EQUITY": MarketVertical.EQUITY_HK.value,
    "HK_EQUITIES": MarketVertical.EQUITY_HK.value,
    "HONG_KONG_EQUITY": MarketVertical.EQUITY_HK.value,
    "CN_FUTURES": MarketVertical.FUTURES_CN.value,
    "CHINA_FUTURES": MarketVertical.FUTURES_CN.value,
    "US_FUTURES": MarketVertical.FUTURES_US.value,
    "US_OPTIONS": MarketVertical.OPTIONS_US.value,
    "CN_OPTIONS": MarketVertical.OPTIONS_CN.value,
    "FX": MarketVertical.FX_SPOT.value,
    "FOREX": MarketVertical.FX_SPOT.value,
    "CRYPTO": MarketVertical.CRYPTO_PERP.value,
}


def normalize_market_vertical(value: Any) -> str:
    """Return a stable uppercase market-vertical identifier."""

    if isinstance(value, MarketVertical):
        return value.value
    if value is None:
        return MarketVertical.UNKNOWN.value

    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not normalized:
        return MarketVertical.UNKNOWN.value
    return _ALIASES.get(normalized, normalized)
