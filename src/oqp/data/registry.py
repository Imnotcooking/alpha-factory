"""Factory helpers for data adapters."""

from __future__ import annotations

from collections.abc import Callable

from oqp.config import OQPSettings, load_settings
from oqp.data.base import FundamentalsAdapter, MarketDataAdapter, OptionsDataAdapter
from oqp.data.vendors import (
    FMPDataAdapter,
    MassiveFlatFilesConfig,
    MassiveOptionsDataAdapter,
    PolygonOptionsSnapshotAdapter,
    YahooDataAdapter,
)


MarketDataFactory = Callable[[OQPSettings], MarketDataAdapter]
FundamentalsFactory = Callable[[OQPSettings], FundamentalsAdapter]
OptionsFactory = Callable[[OQPSettings], OptionsDataAdapter]


def _settings(settings: OQPSettings | None) -> OQPSettings:
    return settings if settings is not None else load_settings()


def _fmp(settings: OQPSettings) -> FMPDataAdapter:
    return FMPDataAdapter(api_key=settings.fmp_api_key)


def _massive_options(settings: OQPSettings) -> MassiveOptionsDataAdapter:
    return MassiveOptionsDataAdapter(
        api_key=settings.massive_api_key or settings.options_api_key,
        flat_files=MassiveFlatFilesConfig(
            access_key_id=settings.massive_flat_files_access_key_id,
            secret_access_key=settings.massive_flat_files_secret_access_key,
            endpoint=settings.massive_flat_files_endpoint,
            bucket=settings.massive_flat_files_bucket,
        ),
    )


_MARKET_DATA_FACTORIES: dict[str, MarketDataFactory] = {
    "fmp": _fmp,
    "yahoo": lambda settings: YahooDataAdapter(),
}

_FUNDAMENTALS_FACTORIES: dict[str, FundamentalsFactory] = {
    "fmp": _fmp,
}

_OPTIONS_FACTORIES: dict[str, OptionsFactory] = {
    "massive": _massive_options,
    "massive_options": _massive_options,
    "options": _massive_options,
    "polygon": lambda settings: PolygonOptionsSnapshotAdapter(
        api_key=(
            settings.massive_api_key
            or settings.options_api_key
            or settings.polygon_api_key
        )
    ),
    "polygon_options": lambda settings: PolygonOptionsSnapshotAdapter(
        api_key=(
            settings.massive_api_key
            or settings.options_api_key
            or settings.polygon_api_key
        )
    ),
}


def register_market_data_adapter(name: str, factory: MarketDataFactory) -> None:
    _MARKET_DATA_FACTORIES[name.lower()] = factory


def register_fundamentals_adapter(name: str, factory: FundamentalsFactory) -> None:
    _FUNDAMENTALS_FACTORIES[name.lower()] = factory


def register_options_adapter(name: str, factory: OptionsFactory) -> None:
    _OPTIONS_FACTORIES[name.lower()] = factory


def get_market_data_adapter(
    name: str = "yahoo", settings: OQPSettings | None = None
) -> MarketDataAdapter:
    key = name.lower()
    if key not in _MARKET_DATA_FACTORIES:
        raise KeyError(f"Unknown market data adapter: {name}")
    return _MARKET_DATA_FACTORIES[key](_settings(settings))


def get_fundamentals_adapter(
    name: str = "fmp", settings: OQPSettings | None = None
) -> FundamentalsAdapter:
    key = name.lower()
    if key not in _FUNDAMENTALS_FACTORIES:
        raise KeyError(f"Unknown fundamentals adapter: {name}")
    return _FUNDAMENTALS_FACTORIES[key](_settings(settings))


def get_options_adapter(
    name: str = "massive", settings: OQPSettings | None = None
) -> OptionsDataAdapter:
    key = name.lower()
    if key not in _OPTIONS_FACTORIES:
        raise KeyError(f"Unknown options adapter: {name}")
    return _OPTIONS_FACTORIES[key](_settings(settings))
