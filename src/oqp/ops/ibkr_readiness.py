#!/usr/bin/env python3
"""Redacted IBKR server readiness checks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


from oqp.brokers import (
    BrokerConnectionConfig,
    BrokerEnvironment,
    BrokerConnectionStatus,
    fetch_ibkr_readonly_portfolio_snapshot,
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import OQPSettings, REPO_ROOT, load_settings
from oqp.portfolio import default_portfolio_ledger_path


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str

    def failed(self) -> bool:
        return self.status == "fail"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether this host is ready for read-only IBKR ingestion.",
    )
    parser.add_argument(
        "--profile",
        choices=("live", "paper"),
        default="live",
        help="IBKR profile to check. Live remains read-only.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to the runtime .env file.",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for the raw TCP socket check.",
    )
    parser.add_argument(
        "--adapter-check",
        action="store_true",
        help="Also connect through the IBKR read-only adapter and fetch account metadata.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env_file)
    checks = run_checks(
        settings,
        profile=args.profile,
        socket_timeout=args.socket_timeout,
        adapter_check=args.adapter_check,
    )

    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        print_checks(checks)

    return 1 if any(check.failed() for check in checks) else 0


def run_checks(
    settings: OQPSettings,
    *,
    profile: str = "live",
    socket_timeout: float = 2.0,
    adapter_check: bool = False,
) -> list[ReadinessCheck]:
    checks: list[ReadinessCheck] = []
    config = intended_config(settings, profile)

    checks.extend(_runtime_checks(settings, config))
    checks.append(_profile_gate_check(settings, profile))
    checks.append(_package_check("ib_insync", required_for="IBKR adapter"))
    checks.append(_package_check("yfinance", required_for="NAV price fetch"))
    checks.append(_ledger_check())
    checks.append(_socket_check(config.host, config.port, timeout=socket_timeout))

    accepted_config = _accepted_profile_config(settings, profile)
    if accepted_config is None:
        checks.append(
            ReadinessCheck(
                "adapter profile",
                "fail",
                "Application safety gate rejects this profile; fix .env before adapter check.",
            )
        )
    elif adapter_check:
        checks.append(_adapter_check(settings, accepted_config))
    else:
        checks.append(
            ReadinessCheck(
                "adapter check",
                "warn",
                "Skipped. Re-run with --adapter-check after IB Gateway/TWS is logged in.",
            )
        )

    return checks


def intended_config(settings: OQPSettings, profile: str) -> BrokerConnectionConfig:
    if profile == "live":
        return BrokerConnectionConfig(
            broker="ibkr",
            host=settings.ibkr_host,
            port=settings.ibkr_live_port,
            client_id=settings.ibkr_live_client_id,
            environment=BrokerEnvironment.LIVE,
            readonly=True,
            metadata={"profile": "ibkr_live_readonly"},
        )
    return BrokerConnectionConfig(
        broker="ibkr",
        host=settings.ibkr_host,
        port=settings.ibkr_paper_port,
        client_id=settings.ibkr_paper_client_id,
        environment=BrokerEnvironment.PAPER,
        readonly=True,
        metadata={"profile": "ibkr_paper_readonly"},
    )


def print_checks(checks: list[ReadinessCheck]) -> None:
    width = max(len(check.name) for check in checks) if checks else 0
    for check in checks:
        print(f"{check.status.upper():5}  {check.name:<{width}}  {check.detail}")


def _runtime_checks(
    settings: OQPSettings,
    config: BrokerConnectionConfig,
) -> list[ReadinessCheck]:
    checks = [
        ReadinessCheck(
            "profile",
            "pass",
            (
                f"{config.metadata.get('profile')} host={config.host} "
                f"port={config.port} client_id={config.client_id}"
            ),
        ),
        ReadinessCheck(
            "read-only config",
            "pass" if config.readonly else "fail",
            "Broker profile is read-only." if config.readonly else "Broker profile is not read-only.",
        ),
        ReadinessCheck(
            "live trading disabled",
            "pass" if not settings.allow_live_trading else "fail",
            (
                "ALLOW_LIVE_TRADING=false."
                if not settings.allow_live_trading
                else "Set ALLOW_LIVE_TRADING=false for middle-office monitoring."
            ),
        ),
    ]

    local_hosts = {"127.0.0.1", "localhost", "::1"}
    checks.append(
        ReadinessCheck(
            "host locality",
            "pass" if config.host in local_hosts else "warn",
            (
                "IBKR socket is local to this server."
                if config.host in local_hosts
                else "Remote IBKR host configured; lock down firewall and Trusted IPs."
            ),
        )
    )
    return checks


def _profile_gate_check(settings: OQPSettings, profile: str) -> ReadinessCheck:
    if profile == "paper":
        return ReadinessCheck(
            "profile gate",
            "pass",
            "Paper read-only profile does not require the live monitor gate.",
        )

    return ReadinessCheck(
        "profile gate",
        "pass" if settings.ibkr_live_monitor_enabled else "fail",
        (
            "IBKR_LIVE_MONITOR_ENABLED=true."
            if settings.ibkr_live_monitor_enabled
            else "Set IBKR_LIVE_MONITOR_ENABLED=true to allow live read-only monitoring."
        ),
    )


def _package_check(package: str, *, required_for: str) -> ReadinessCheck:
    found = importlib.util.find_spec(package) is not None
    return ReadinessCheck(
        f"python package {package}",
        "pass" if found else "fail",
        f"Installed for {required_for}." if found else f"Missing package for {required_for}.",
    )


def _ledger_check() -> ReadinessCheck:
    path = default_portfolio_ledger_path()
    parent = path.parent
    if path.exists():
        return ReadinessCheck("portfolio ledger", "pass", f"Found {path}.")
    if parent.exists():
        return ReadinessCheck(
            "portfolio ledger",
            "warn",
            f"Ledger not found yet, but parent exists: {parent}.",
        )
    return ReadinessCheck(
        "portfolio ledger",
        "warn",
        f"Ledger parent does not exist yet: {parent}. ETL can create it.",
    )


def _socket_check(host: str, port: int, *, timeout: float) -> ReadinessCheck:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except OSError as exc:
        return ReadinessCheck(
            "ibkr socket",
            "fail",
            f"Cannot connect to {host}:{port}: {exc}",
        )
    return ReadinessCheck(
        "ibkr socket",
        "pass",
        f"TCP connection accepted on {host}:{port}.",
    )


def _accepted_profile_config(
    settings: OQPSettings,
    profile: str,
) -> BrokerConnectionConfig | None:
    try:
        return get_broker_profile_config(
            "ibkr_live_readonly" if profile == "live" else "ibkr_paper_readonly",
            settings=settings,
        )
    except Exception:
        return None


def _adapter_check(
    settings: OQPSettings,
    config: BrokerConnectionConfig,
) -> ReadinessCheck:
    try:
        broker = get_broker_adapter("ibkr", settings=settings)
        snapshot = fetch_ibkr_readonly_portfolio_snapshot(config, adapter=broker)
    except Exception as exc:
        return ReadinessCheck("adapter check", "fail", f"Adapter failed: {exc}")

    if snapshot.health.status != BrokerConnectionStatus.CONNECTED or snapshot.error:
        return ReadinessCheck(
            "adapter check",
            "fail",
            snapshot.error or snapshot.health.message or "Adapter did not connect.",
        )

    account = _redact_account(snapshot.health.account_id)
    cash = snapshot.metrics.get("Available_Cash_USD")
    nav = snapshot.metrics.get("Total_NAV_USD")
    return ReadinessCheck(
        "adapter check",
        "pass",
        (
            f"Connected read-only. account={account} "
            f"positions={len(snapshot.position_rows)} "
            f"cash={_moneyish(cash)} nav={_moneyish(nav)}"
        ),
    )


def _redact_account(account_id: str | None) -> str:
    if not account_id:
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"


def _moneyish(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
