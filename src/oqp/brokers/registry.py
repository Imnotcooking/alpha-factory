"""Factory helpers for broker adapters."""

from __future__ import annotations

from collections.abc import Callable

from oqp.brokers.base import BrokerAdapter
from oqp.brokers.ibkr import IBKRBrokerAdapter
from oqp.brokers.models import BrokerConnectionConfig, BrokerEnvironment
from oqp.config import OQPSettings, load_settings


BrokerFactory = Callable[[OQPSettings], BrokerAdapter]


class BrokerProfileError(ValueError):
    """Raised when a requested broker profile violates runtime safety settings."""


def _settings(settings: OQPSettings | None) -> OQPSettings:
    return settings if settings is not None else load_settings()


_BROKER_FACTORIES: dict[str, BrokerFactory] = {
    "ibkr": lambda settings: IBKRBrokerAdapter(),
    "interactive_brokers": lambda settings: IBKRBrokerAdapter(),
}


def register_broker_adapter(name: str, factory: BrokerFactory) -> None:
    _BROKER_FACTORIES[name.lower()] = factory


def get_broker_adapter(
    name: str = "ibkr", settings: OQPSettings | None = None
) -> BrokerAdapter:
    key = name.lower()
    if key not in _BROKER_FACTORIES:
        raise KeyError(f"Unknown broker adapter: {name}")
    return _BROKER_FACTORIES[key](_settings(settings))


def get_broker_connection_config(
    name: str = "ibkr",
    settings: OQPSettings | None = None,
    environment: BrokerEnvironment | None = None,
    readonly: bool = True,
) -> BrokerConnectionConfig:
    active_settings = _settings(settings)
    broker_environment = environment or BrokerEnvironment(
        active_settings.trading_mode.lower()
        if active_settings.trading_mode.lower() == "live"
        else "paper"
    )

    return BrokerConnectionConfig(
        broker=name,
        host=active_settings.ibkr_host,
        port=(
            active_settings.ibkr_live_port
            if broker_environment == BrokerEnvironment.LIVE
            else active_settings.ibkr_paper_port
        ),
        client_id=active_settings.ibkr_client_id,
        environment=broker_environment,
        readonly=readonly,
    )


def get_broker_profile_config(
    profile: str = "ibkr_paper",
    settings: OQPSettings | None = None,
) -> BrokerConnectionConfig:
    active_settings = _settings(settings)
    key = profile.lower()

    if key in {"ibkr_paper", "ibkr_paper_readonly", "paper", "paper_readonly"}:
        return BrokerConnectionConfig(
            broker="ibkr",
            host=active_settings.ibkr_host,
            port=active_settings.ibkr_paper_port,
            client_id=active_settings.ibkr_paper_client_id,
            environment=BrokerEnvironment.PAPER,
            readonly=True,
            metadata={"profile": "ibkr_paper_readonly"},
        )

    if key in {"ibkr_paper_submit", "paper_submit"}:
        if not active_settings.allow_paper_order_submit:
            raise BrokerProfileError(
                "IBKR paper submit profile is disabled. Set "
                "ALLOW_PAPER_ORDER_SUBMIT=true only for guarded paper execution."
            )
        if active_settings.allow_live_trading:
            raise BrokerProfileError(
                "Paper submit profile requires ALLOW_LIVE_TRADING=false."
            )
        return BrokerConnectionConfig(
            broker="ibkr",
            host=active_settings.ibkr_host,
            port=active_settings.ibkr_paper_port,
            client_id=active_settings.ibkr_paper_submit_client_id,
            environment=BrokerEnvironment.PAPER,
            readonly=False,
            metadata={"profile": "ibkr_paper_submit"},
        )

    if key in {"ibkr_live_readonly", "live_readonly", "live_monitor"}:
        if not active_settings.ibkr_live_monitor_enabled:
            raise BrokerProfileError(
                "IBKR live read-only monitor is disabled. Set "
                "IBKR_LIVE_MONITOR_ENABLED=true to enable this profile."
            )
        return BrokerConnectionConfig(
            broker="ibkr",
            host=active_settings.ibkr_host,
            port=active_settings.ibkr_live_port,
            client_id=active_settings.ibkr_live_client_id,
            environment=BrokerEnvironment.LIVE,
            readonly=True,
            metadata={"profile": "ibkr_live_readonly"},
        )

    if key in {"ibkr_live", "live"}:
        if not active_settings.allow_live_trading:
            raise BrokerProfileError(
                "Live trading profile is disabled. Set ALLOW_LIVE_TRADING=true "
                "only after execution safety controls are complete."
            )
        return BrokerConnectionConfig(
            broker="ibkr",
            host=active_settings.ibkr_host,
            port=active_settings.ibkr_live_port,
            client_id=active_settings.ibkr_live_client_id,
            environment=BrokerEnvironment.LIVE,
            readonly=False,
            metadata={"profile": "ibkr_live"},
        )

    raise KeyError(f"Unknown broker profile: {profile}")
