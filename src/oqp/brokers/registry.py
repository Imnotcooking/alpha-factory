"""Factory helpers for broker adapters."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable
from urllib.parse import urlparse

from oqp.brokers.base import BrokerAdapter
from oqp.brokers.ibkr import IBKRBrokerAdapter
from oqp.brokers.models import BrokerConnectionConfig, BrokerEnvironment
from oqp.brokers.qmt import QMTBrokerAdapter
from oqp.config import OQPSettings, load_settings


BrokerFactory = Callable[[OQPSettings], BrokerAdapter]
TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class BrokerProfileError(ValueError):
    """Raised when a requested broker profile violates runtime safety settings."""


def _settings(settings: OQPSettings | None) -> OQPSettings:
    return settings if settings is not None else load_settings()


_BROKER_FACTORIES: dict[str, BrokerFactory] = {
    "ibkr": lambda settings: IBKRBrokerAdapter(),
    "interactive_brokers": lambda settings: IBKRBrokerAdapter(),
    "qmt": lambda settings: QMTBrokerAdapter(timeout=settings.qmt_timeout_seconds),
    "xtquant": lambda settings: QMTBrokerAdapter(timeout=settings.qmt_timeout_seconds),
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

    if key in {"qmt_paper", "qmt_paper_readonly"}:
        _validate_qmt_connector_url(
            active_settings,
            active_settings.qmt_connector_url,
            "QMT_CONNECTOR_URL",
        )
        host, port = _qmt_connector_host_port(
            active_settings,
            active_settings.qmt_connector_url,
        )
        return BrokerConnectionConfig(
            broker="qmt",
            host=host,
            port=port,
            client_id=active_settings.qmt_session_id,
            environment=BrokerEnvironment.PAPER,
            account_id=active_settings.qmt_paper_account_id
            or active_settings.qmt_account_id,
            readonly=True,
            metadata=_qmt_metadata(active_settings, "qmt_paper_readonly"),
        )

    if key in {"qmt_paper_submit", "qmt_submit"}:
        if not active_settings.allow_paper_order_submit:
            raise BrokerProfileError(
                "QMT paper submit requires ALLOW_PAPER_ORDER_SUBMIT=true."
            )
        if not active_settings.allow_qmt_paper_order_submit:
            raise BrokerProfileError(
                "QMT paper submit requires ALLOW_QMT_PAPER_ORDER_SUBMIT=true."
            )
        if active_settings.allow_live_trading:
            raise BrokerProfileError(
                "QMT paper submit requires ALLOW_LIVE_TRADING=false."
            )
        _validate_qmt_submit_security(active_settings, "QMT paper submit")
        _validate_qmt_connector_url(
            active_settings,
            active_settings.qmt_submit_connector_url,
            "QMT_SUBMIT_CONNECTOR_URL",
        )
        host, port = _qmt_connector_host_port(
            active_settings,
            active_settings.qmt_submit_connector_url,
        )
        return BrokerConnectionConfig(
            broker="qmt",
            host=host,
            port=port,
            client_id=active_settings.qmt_session_id,
            environment=BrokerEnvironment.PAPER,
            account_id=active_settings.qmt_paper_account_id
            or active_settings.qmt_account_id,
            readonly=False,
            metadata=_qmt_metadata(
                active_settings,
                "qmt_paper_submit",
                connector_url=active_settings.qmt_submit_connector_url,
            ),
        )

    if key in {"qmt_live_readonly", "qmt_live_monitor"}:
        if not active_settings.qmt_live_monitor_enabled:
            raise BrokerProfileError(
                "QMT live read-only monitor is disabled. Set "
                "QMT_LIVE_MONITOR_ENABLED=true to enable this profile."
            )
        _validate_qmt_connector_url(
            active_settings,
            active_settings.qmt_connector_url,
            "QMT_CONNECTOR_URL",
        )
        host, port = _qmt_connector_host_port(
            active_settings,
            active_settings.qmt_connector_url,
        )
        return BrokerConnectionConfig(
            broker="qmt",
            host=host,
            port=port,
            client_id=active_settings.qmt_session_id,
            environment=BrokerEnvironment.LIVE,
            account_id=active_settings.qmt_live_account_id
            or active_settings.qmt_account_id,
            readonly=True,
            metadata=_qmt_metadata(active_settings, "qmt_live_readonly"),
        )

    if key in {"qmt_live", "qmt_live_submit"}:
        if not active_settings.allow_live_trading:
            raise BrokerProfileError(
                "QMT live submit requires ALLOW_LIVE_TRADING=true."
            )
        if not active_settings.allow_qmt_live_trading:
            raise BrokerProfileError(
                "QMT live submit requires ALLOW_QMT_LIVE_TRADING=true."
            )
        _validate_qmt_submit_security(active_settings, "QMT live submit")
        _validate_qmt_connector_url(
            active_settings,
            active_settings.qmt_submit_connector_url,
            "QMT_SUBMIT_CONNECTOR_URL",
        )
        host, port = _qmt_connector_host_port(
            active_settings,
            active_settings.qmt_submit_connector_url,
        )
        return BrokerConnectionConfig(
            broker="qmt",
            host=host,
            port=port,
            client_id=active_settings.qmt_session_id,
            environment=BrokerEnvironment.LIVE,
            account_id=active_settings.qmt_live_account_id
            or active_settings.qmt_account_id,
            readonly=False,
            metadata={
                **_qmt_metadata(
                    active_settings,
                    "qmt_live_submit",
                    connector_url=active_settings.qmt_submit_connector_url,
                ),
                "allow_qmt_live_trading": True,
            },
        )

    raise KeyError(f"Unknown broker profile: {profile}")


def _qmt_connector_host_port(settings: OQPSettings, connector_url: str) -> tuple[str, int]:
    parsed = urlparse(connector_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _qmt_metadata(
    settings: OQPSettings,
    profile: str,
    *,
    connector_url: str | None = None,
) -> dict[str, object]:
    active_connector_url = connector_url or settings.qmt_connector_url
    return {
        "profile": profile,
        "connector_url": active_connector_url,
        "read_connector_url": settings.qmt_connector_url,
        "submit_connector_url": settings.qmt_submit_connector_url,
        "connector_enabled": settings.qmt_connector_enabled,
        "api_token": settings.qmt_api_token,
        "request_signing_secret": settings.qmt_request_signing_secret,
        "audit_log_path": str(settings.qmt_audit_log_path),
        "require_private_connector": settings.qmt_require_private_connector,
        "account_type": settings.qmt_account_type,
        "session_id": settings.qmt_session_id,
    }


def _validate_qmt_submit_security(settings: OQPSettings, label: str) -> None:
    if not settings.qmt_api_token:
        raise BrokerProfileError(f"{label} requires QMT_API_TOKEN.")
    if not settings.qmt_request_signing_secret:
        raise BrokerProfileError(f"{label} requires QMT_REQUEST_SIGNING_SECRET.")
    read_url = settings.qmt_connector_url.rstrip("/")
    submit_url = settings.qmt_submit_connector_url.rstrip("/")
    if submit_url == read_url:
        raise BrokerProfileError(
            f"{label} requires QMT_SUBMIT_CONNECTOR_URL to be separate from QMT_CONNECTOR_URL."
        )


def _validate_qmt_connector_url(
    settings: OQPSettings,
    connector_url: str,
    label: str,
) -> None:
    if not settings.qmt_require_private_connector:
        return
    parsed = urlparse(connector_url)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise BrokerProfileError(f"{label} must include a hostname.")
    if _is_private_connector_host(host):
        return
    raise BrokerProfileError(
        f"{label} must point to localhost, a private LAN IP, or a Tailscale/WireGuard address. "
        "Set QMT_REQUIRE_PRIVATE_CONNECTOR=false only for a reviewed tunnel/proxy exception."
    )


def _is_private_connector_host(host: str) -> bool:
    if host in {"localhost"} or host.endswith(".local") or host.endswith(".ts.net"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if ip.is_unspecified:
        return False
    return bool(ip.is_loopback or ip.is_private or ip in TAILSCALE_CGNAT)
