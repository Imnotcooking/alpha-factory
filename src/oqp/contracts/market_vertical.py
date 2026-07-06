"""Market vertical names shared by research, trading, and dashboards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class MarketVerticalSpec:
    vertical: str
    description: str
    region: str
    t_settlement: int
    price_limit: bool
    vectorizable: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("vertical", None)
        return payload


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


MARKET_VERTICAL_SPECS: dict[str, MarketVerticalSpec] = {
    MarketVertical.EQUITY_US.value: MarketVerticalSpec(
        vertical=MarketVertical.EQUITY_US.value,
        description="US Equities (NYSE/NASDAQ)",
        region="US",
        t_settlement=0,
        price_limit=False,
        vectorizable=True,
    ),
    MarketVertical.EQUITY_CN.value: MarketVerticalSpec(
        vertical=MarketVertical.EQUITY_CN.value,
        description="China A-Shares (SHFE/SZSE)",
        region="CN",
        t_settlement=1,
        price_limit=True,
        vectorizable=True,
    ),
    MarketVertical.EQUITY_HK.value: MarketVerticalSpec(
        vertical=MarketVertical.EQUITY_HK.value,
        description="Hong Kong Equities (HKEX)",
        region="HK",
        t_settlement=0,
        price_limit=False,
        vectorizable=True,
    ),
    MarketVertical.FUTURES_US.value: MarketVerticalSpec(
        vertical=MarketVertical.FUTURES_US.value,
        description="US Futures (CME/CBOT)",
        region="US",
        t_settlement=0,
        price_limit=False,
        vectorizable=True,
    ),
    MarketVertical.FUTURES_CN.value: MarketVerticalSpec(
        vertical=MarketVertical.FUTURES_CN.value,
        description="Chinese Futures (DCE/CZCE/SHFE/CFFEX)",
        region="CN",
        t_settlement=0,
        price_limit=True,
        vectorizable=True,
    ),
    MarketVertical.OPTIONS_US.value: MarketVerticalSpec(
        vertical=MarketVertical.OPTIONS_US.value,
        description="US Options",
        region="US",
        t_settlement=1,
        price_limit=False,
        vectorizable=False,
    ),
    MarketVertical.FX_SPOT.value: MarketVerticalSpec(
        vertical=MarketVertical.FX_SPOT.value,
        description="Global Foreign Exchange (Spot)",
        region="GLOBAL",
        t_settlement=0,
        price_limit=False,
        vectorizable=True,
    ),
    MarketVertical.CRYPTO_PERP.value: MarketVerticalSpec(
        vertical=MarketVertical.CRYPTO_PERP.value,
        description="Crypto Perpetual Swaps",
        region="GLOBAL",
        t_settlement=0,
        price_limit=False,
        vectorizable=True,
    ),
}

ASSET_TAXONOMY: dict[str, dict[str, Any]] = {
    key: spec.to_dict() for key, spec in MARKET_VERTICAL_SPECS.items()
}


def normalize_market_vertical(value: Any) -> str:
    """Return a stable uppercase market-vertical identifier."""

    if isinstance(value, MarketVertical):
        return value.value
    if value is None:
        return MarketVertical.UNKNOWN.value
    try:
        if value != value:
            return MarketVertical.UNKNOWN.value
    except Exception:
        pass

    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    if not normalized:
        return MarketVertical.UNKNOWN.value
    return _ALIASES.get(normalized, normalized)


def market_vertical_spec(value: Any) -> MarketVerticalSpec | None:
    """Return metadata for a normalized market vertical."""

    return MARKET_VERTICAL_SPECS.get(normalize_market_vertical(value))


def market_vertical_taxonomy() -> dict[str, dict[str, Any]]:
    """Return a copy of the public market-vertical taxonomy."""

    return {key: dict(value) for key, value in ASSET_TAXONOMY.items()}
