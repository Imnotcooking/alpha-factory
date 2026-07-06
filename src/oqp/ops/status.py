"""Status collection for the Ops Dashboard.

The functions here avoid hard dependencies on systemd, Docker, or Linux-only
APIs so the dashboard can still render locally on macOS during development.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.accounts import (
    default_account_ledger_path,
    ensure_account_ledger_schema,
    load_account_trade_events,
)
from oqp.brokers import (
    BrokerConnectionConfig,
    BrokerConnectionStatus,
    fetch_ibkr_readonly_portfolio_snapshot,
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import REPO_ROOT, OQPSettings, load_settings


DEFAULT_PORTFOLIO_HEALTH_PATH = REPO_ROOT / "runtime" / "logs" / "portfolio_snapshot_health.json"
DEFAULT_PAPER_HEALTH_PATH = REPO_ROOT / "runtime" / "logs" / "paper_trading_health.json"
DEFAULT_IBKR_HEARTBEAT_HEALTH_PATH = (
    REPO_ROOT / "runtime" / "logs" / "ibkr_adapter_heartbeat_health.json"
)
DEFAULT_SERVER_IBKR_READINESS_PATH = (
    REPO_ROOT / "runtime" / "logs" / "server_ibkr_readiness_health.json"
)
IBKR_HEARTBEAT_CLIENT_ID_OFFSET = 9_000
SYSTEMD_UNITS = (
    "oqp-research-dashboard.service",
    "oqp-ops-dashboard.service",
    "oqp-money-dashboard.service",
    "oqp-paper-dashboard.service",
    "oqp-portfolio-snapshot.timer",
    "oqp-paper-snapshot.timer",
    "oqp-paper-strategy-runner.timer",
)


@dataclass(frozen=True, slots=True)
class OpsStatusItem:
    category: str
    name: str
    status: str
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    @property
    def warned(self) -> bool:
        return self.status == "warn"

    def to_row(self) -> dict[str, Any]:
        row = {
            "Category": self.category,
            "Check": self.name,
            "Status": self.status,
            "Detail": self.detail,
        }
        row.update(self.metadata)
        return row


@dataclass(frozen=True, slots=True)
class OpsStatusSnapshot:
    checked_at: datetime
    items: tuple[OpsStatusItem, ...]
    account_rows: tuple[dict[str, Any], ...]
    event_rows: tuple[dict[str, Any], ...]
    host_summary: dict[str, Any]

    @property
    def overall_status(self) -> str:
        if any(item.failed for item in self.items):
            return "fail"
        if any(item.warned for item in self.items):
            return "warn"
        return "pass"

    @property
    def item_rows(self) -> list[dict[str, Any]]:
        return [item.to_row() for item in self.items]


def collect_ops_status(
    *,
    settings: OQPSettings | None = None,
    account_ledger_path: str | Path | None = None,
    portfolio_health_path: str | Path = DEFAULT_PORTFOLIO_HEALTH_PATH,
    paper_health_path: str | Path = DEFAULT_PAPER_HEALTH_PATH,
    ibkr_heartbeat_health_path: str | Path = DEFAULT_IBKR_HEARTBEAT_HEALTH_PATH,
    server_ibkr_readiness_path: str | Path = DEFAULT_SERVER_IBKR_READINESS_PATH,
    max_age_hours: float = 36.0,
    repo_root: str | Path = REPO_ROOT,
) -> OpsStatusSnapshot:
    active_settings = settings or load_settings()
    ledger_path = Path(account_ledger_path) if account_ledger_path else default_account_ledger_path()
    root = Path(repo_root)

    account_rows = latest_account_rows(ledger_path)
    event_rows = latest_account_event_rows(ledger_path)
    items: list[OpsStatusItem] = []
    status_source = active_settings.ops_status_source
    items.append(
        OpsStatusItem(
            "Gateway",
            "Broker status source",
            "pass",
            (
                "Using synced server health files and account ledgers."
                if status_source == "snapshot"
                else "Using direct local broker socket/API checks."
            ),
            {"Source": status_source},
        )
    )
    if status_source == "snapshot":
        items.extend(
            health_file_items(
                Path(ibkr_heartbeat_health_path),
                label="IBKR Adapter Heartbeat",
                category="Broker Heartbeat",
                max_age_hours=max_age_hours,
            )
        )
        items.extend(
            server_ibkr_readiness_items(
                Path(server_ibkr_readiness_path),
                max_age_hours=max_age_hours,
            )
        )
    else:
        items.extend(
            [
                socket_status_item(
                    "Live IBKR Gateway",
                    active_settings.ibkr_host,
                    active_settings.ibkr_live_port,
                    category="Gateway",
                ),
                socket_status_item(
                    "Paper IBKR Gateway",
                    active_settings.ibkr_host,
                    active_settings.ibkr_paper_port,
                    category="Gateway",
                ),
            ]
        )
        items.extend(ibkr_api_heartbeat_items(active_settings))
    items.extend(account_freshness_items(account_rows, max_age_hours=max_age_hours))
    items.extend(account_event_items(event_rows))
    items.extend(
        health_file_items(
            Path(portfolio_health_path),
            label="Portfolio Snapshot",
            category="Jobs",
            max_age_hours=max_age_hours,
        )
    )
    items.extend(
        health_file_items(
            Path(paper_health_path),
            label="Paper Snapshot",
            category="Jobs",
            max_age_hours=max_age_hours,
        )
    )
    if status_source != "snapshot":
        items.extend(
            health_file_items(
                Path(ibkr_heartbeat_health_path),
                label="IBKR Adapter Heartbeat",
                category="Jobs",
                max_age_hours=max_age_hours,
            )
        )
    if status_source == "snapshot":
        items.extend(snapshot_scheduler_items())
        items.extend(snapshot_notification_items())
        items.extend(safety_status_items(active_settings, snapshot_mode=True))
    else:
        items.extend(systemd_status_items())
        items.extend(cron_status_items())
        items.extend(discord_status_items())
        items.extend(safety_status_items(active_settings))
    host_summary = host_summary_values(root)
    items.extend(host_health_items(host_summary, snapshot_mode=status_source == "snapshot"))

    return OpsStatusSnapshot(
        checked_at=datetime.now(timezone.utc),
        items=tuple(items),
        account_rows=tuple(account_rows),
        event_rows=tuple(event_rows),
        host_summary=host_summary,
    )


def latest_account_rows(db_path: str | Path) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []

    ensure_account_ledger_schema(path)
    query = """
        SELECT n.date, n.account_key, n.account_id, n.broker, n.profile,
               n.environment, n.as_of, n.net_liquidation, n.cash, n.daily_pnl,
               n.position_count, n.snapshot_id
        FROM account_nav n
        JOIN (
            SELECT account_key, MAX(as_of) AS max_as_of
            FROM account_nav
            GROUP BY account_key
        ) latest
          ON n.account_key = latest.account_key AND n.as_of = latest.max_as_of
        ORDER BY n.environment, n.profile
    """
    with sqlite3.connect(path) as conn:
        frame = pd.read_sql(query, conn)
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "date": row.get("date"),
                "account_key": row.get("account_key"),
                "account_id": _redact_account(row.get("account_id")),
                "broker": row.get("broker"),
                "profile": row.get("profile"),
                "environment": row.get("environment"),
                "as_of": row.get("as_of"),
                "net_liquidation": _float(row.get("net_liquidation")),
                "cash": _float(row.get("cash")),
                "daily_pnl": _float(row.get("daily_pnl")),
                "position_count": int(row.get("position_count") or 0),
                "snapshot_id": row.get("snapshot_id"),
                "age_hours": _datetime_age_hours(row.get("as_of")),
            }
        )
    return rows


def latest_account_event_rows(db_path: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        return []

    frame = load_account_trade_events(path, limit=limit)
    if frame.empty:
        return []

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "occurred_at": row.get("occurred_at"),
                "environment": row.get("environment"),
                "profile": row.get("profile"),
                "account_id": _redact_account(row.get("account_id")),
                "event_type": row.get("event_type"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "quantity": _float(row.get("quantity")),
                "price": _float(row.get("price")),
                "strategy_id": row.get("strategy_id"),
                "order_id": row.get("order_id"),
                "broker_order_id": row.get("broker_order_id"),
            }
        )
    return rows


def account_freshness_items(
    rows: list[dict[str, Any]],
    *,
    max_age_hours: float,
) -> list[OpsStatusItem]:
    if not rows:
        return [
            OpsStatusItem(
                "Accounts",
                "Unified account ledger",
                "warn",
                "No account_nav rows found yet.",
            )
        ]

    items: list[OpsStatusItem] = []
    by_environment = {str(row.get("environment")): row for row in rows}
    for environment in ("live", "paper"):
        row = by_environment.get(environment)
        if row is None:
            items.append(
                OpsStatusItem(
                    "Accounts",
                    f"{environment.title()} account snapshot",
                    "warn",
                    "No latest account row found.",
                )
            )
            continue
        age_hours = row.get("age_hours")
        is_fresh = age_hours is not None and age_hours <= max_age_hours
        status = "pass" if is_fresh else "warn"
        age_text = "unknown age" if age_hours is None else f"{float(age_hours):.1f}h old"
        detail = (
            f"{'fresh' if is_fresh else 'stale'} snapshot ({age_text}): "
            f"as_of={row.get('as_of')} nav={_money(row.get('net_liquidation'))} "
            f"positions={row.get('position_count')}"
        )
        items.append(
            OpsStatusItem(
                "Accounts",
                f"{environment.title()} account snapshot",
                status,
                detail,
                {
                    "Age Hours": "" if age_hours is None else round(float(age_hours), 2),
                    "Profile": str(row.get("profile") or ""),
                },
            )
        )
    return items


def account_event_items(rows: list[dict[str, Any]]) -> list[OpsStatusItem]:
    if not rows:
        return [
            OpsStatusItem(
                "Accounts",
                "Account trade events",
                "pass",
                "No account trade events recorded yet.",
            )
        ]

    latest = rows[0]
    environments = sorted({str(row.get("environment") or "") for row in rows})
    return [
        OpsStatusItem(
            "Accounts",
            "Account trade events",
            "pass",
            (
                f"events={len(rows)} latest={latest.get('event_type')} "
                f"{latest.get('symbol')} at {latest.get('occurred_at')}"
            ),
            {"Environments": ", ".join(item for item in environments if item)},
        )
    ]


def socket_status_item(
    name: str,
    host: str,
    port: int,
    *,
    category: str = "Gateway",
    timeout: float = 1.0,
) -> OpsStatusItem:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            pass
    except OSError as exc:
        return OpsStatusItem(
            category,
            name,
            "fail",
            f"{host}:{port} unreachable: {exc}",
            {"Host": host, "Port": int(port)},
        )
    return OpsStatusItem(
        category,
        name,
        "pass",
        f"{host}:{port} accepted TCP connection.",
        {"Host": host, "Port": int(port)},
    )


def ibkr_api_heartbeat_items(settings: OQPSettings) -> list[OpsStatusItem]:
    """Perform read-only IBKR API handshakes, beyond the raw socket probes."""

    return [
        ibkr_api_heartbeat_item(
            "Live IBKR API heartbeat",
            "ibkr_live_readonly",
            settings,
        ),
        ibkr_api_heartbeat_item(
            "Paper IBKR API heartbeat",
            "ibkr_paper_readonly",
            settings,
        ),
    ]


def ibkr_api_heartbeat_item(
    name: str,
    profile: str,
    settings: OQPSettings,
) -> OpsStatusItem:
    try:
        config = get_broker_profile_config(profile, settings=settings)
    except Exception as exc:
        return OpsStatusItem(
            "Broker Heartbeat",
            name,
            "warn",
            f"Skipped: {exc}",
            {"Profile": profile},
        )

    heartbeat_config = _heartbeat_config(config)
    try:
        broker = get_broker_adapter("ibkr", settings=settings)
        snapshot = fetch_ibkr_readonly_portfolio_snapshot(
            heartbeat_config,
            adapter=broker,
        )
    except Exception as exc:
        return OpsStatusItem(
            "Broker Heartbeat",
            name,
            "fail",
            f"Adapter exception: {exc}",
            _heartbeat_metadata(heartbeat_config),
        )

    metadata = _heartbeat_metadata(heartbeat_config)
    metadata["Account"] = _redact_account(snapshot.health.account_id)
    metadata["Positions"] = len(snapshot.position_rows)

    if snapshot.health.status != BrokerConnectionStatus.CONNECTED or snapshot.error:
        return OpsStatusItem(
            "Broker Heartbeat",
            name,
            "fail",
            snapshot.error or snapshot.health.message or "Adapter did not connect.",
            metadata,
        )

    cash = _money(snapshot.metrics.get("Available_Cash_USD"))
    nav = _money(snapshot.metrics.get("Total_NAV_USD"))
    return OpsStatusItem(
        "Broker Heartbeat",
        name,
        "pass",
        (
            f"Connected read-only. account={metadata['Account']} "
            f"positions={metadata['Positions']} cash={cash} nav={nav}"
        ),
        metadata,
    )


def _heartbeat_config(
    config: BrokerConnectionConfig,
    *,
    process_id: int | None = None,
) -> BrokerConnectionConfig:
    pid_component = (os.getpid() if process_id is None else process_id) % 100_000
    return replace(
        config,
        client_id=int(config.client_id) + IBKR_HEARTBEAT_CLIENT_ID_OFFSET + pid_component,
    )


def _heartbeat_metadata(config: BrokerConnectionConfig) -> dict[str, Any]:
    return {
        "Profile": str(config.metadata.get("profile") or ""),
        "Host": config.host,
        "Port": config.port,
        "Client ID": config.client_id,
        "Environment": config.environment.value,
        "Read Only": config.readonly,
    }


def health_file_items(
    path: Path,
    *,
    label: str,
    category: str,
    max_age_hours: float,
) -> list[OpsStatusItem]:
    if not path.exists():
        return [OpsStatusItem(category, label, "warn", f"Missing {path}.")]

    age_hours = _file_age_hours(path)
    items = [
        OpsStatusItem(
            category,
            f"{label} status file",
            "pass" if age_hours <= max_age_hours else "warn",
            f"Modified {age_hours:.1f} hours ago at {_mtime(path).isoformat()}.",
            {"Age Hours": round(age_hours, 2)},
        )
    ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        items.append(OpsStatusItem(category, f"{label} payload", "warn", str(exc)))
        return items

    status = str(payload.get("status", "unknown"))
    checked_at = str(payload.get("checked_at", ""))
    items.append(
        OpsStatusItem(
            category,
            f"{label} checks",
            "pass" if status == "pass" else "fail",
            f"status={status} checked_at={checked_at}",
        )
    )
    return items


def server_ibkr_readiness_items(
    path: Path,
    *,
    max_age_hours: float,
) -> list[OpsStatusItem]:
    if not path.exists():
        return [
            OpsStatusItem(
                "Broker Heartbeat",
                "Server IBKR readiness",
                "warn",
                f"Missing {path}. Run scripts/sync_server_runtime.sh to refresh server readiness.",
            )
        ]

    age_hours = _file_age_hours(path)
    items = [
        OpsStatusItem(
            "Broker Heartbeat",
            "Server IBKR readiness status file",
            "pass" if age_hours <= max_age_hours else "warn",
            f"Modified {age_hours:.1f} hours ago at {_mtime(path).isoformat()}.",
            {"Age Hours": round(age_hours, 2)},
        )
    ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        items.append(OpsStatusItem("Broker Heartbeat", "Server IBKR readiness payload", "warn", str(exc)))
        return items

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        items.append(
            OpsStatusItem(
                "Broker Heartbeat",
                "Server IBKR readiness payload",
                "warn",
                "No profile readiness data found.",
            )
        )
        return items

    for profile in ("live", "paper"):
        profile_payload = profiles.get(profile, {})
        if not isinstance(profile_payload, dict):
            continue
        checks = profile_payload.get("checks", [])
        adapter_detail = ""
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict) and check.get("name") == "adapter check":
                    adapter_detail = str(check.get("detail") or "")
                    break
        status = str(profile_payload.get("status") or "warn")
        detail = adapter_detail or f"status={status} checked_at={payload.get('checked_at', '')}"
        items.append(
            OpsStatusItem(
                "Broker Heartbeat",
                f"{profile.title()} IBKR server readiness",
                status if status in {"pass", "warn", "fail"} else "warn",
                detail,
                {
                    "Profile": profile,
                    "Age Hours": round(age_hours, 2),
                },
            )
        )
    return items


def systemd_status_items() -> list[OpsStatusItem]:
    if shutil.which("systemctl") is None:
        return [
            OpsStatusItem(
                "Schedulers",
                "systemd",
                "warn",
                "systemctl is unavailable on this host.",
            )
        ]

    items: list[OpsStatusItem] = []
    for unit in SYSTEMD_UNITS:
        result = command_status(
            ["systemctl", "show", unit, "-p", "LoadState", "-p", "ActiveState", "-p", "SubState", "-p", "Result", "--no-pager"],
            timeout=3.0,
        )
        if result["status"] != "pass":
            items.append(
                OpsStatusItem(
                    "Schedulers",
                    unit,
                    "warn",
                    result["detail"],
                )
            )
            continue
        props = _systemd_props(result["stdout"])
        active = props.get("ActiveState", "unknown")
        load_state = props.get("LoadState", "unknown")
        service_result = props.get("Result", "")
        is_timer = unit.endswith(".timer")
        ok = load_state == "loaded" and active in {"active", "inactive"}
        if is_timer:
            ok = load_state == "loaded" and active == "active"
        if service_result and service_result not in {"success", ""}:
            ok = False
        items.append(
            OpsStatusItem(
                "Schedulers",
                unit,
                "pass" if ok else "warn",
                f"load={load_state} active={active} sub={props.get('SubState', 'unknown')} result={service_result or 'n/a'}",
            )
        )
    return items


def cron_status_items() -> list[OpsStatusItem]:
    if shutil.which("crontab") is None:
        return [
            OpsStatusItem(
                "Schedulers",
                "cron",
                "warn",
                "crontab is unavailable on this host.",
            )
        ]
    result = command_status(["crontab", "-l"], timeout=3.0, allow_codes=(0, 1))
    if result["status"] == "fail":
        return [OpsStatusItem("Schedulers", "cron", "warn", result["detail"])]
    lines = [
        line
        for line in result["stdout"].splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    oqp_lines = [
        line
        for line in lines
        if "run_portfolio_snapshot_job.sh" in line or "run_paper_snapshot_job.sh" in line
    ]
    return [
        OpsStatusItem(
            "Schedulers",
            "cron snapshot jobs",
            "pass" if not oqp_lines else "warn",
            (
                "No Alpha Factory cron snapshot jobs active; systemd timers own scheduling."
                if not oqp_lines
                else f"{len(oqp_lines)} Alpha Factory cron snapshot job(s) still active."
            ),
            {"Cron Lines": len(oqp_lines)},
        )
    ]


def snapshot_scheduler_items() -> list[OpsStatusItem]:
    return [
        OpsStatusItem(
            "Schedulers",
            "Server-owned scheduling",
            "pass",
            "Local dashboard is in snapshot mode; scheduling is verified through synced server health files.",
        )
    ]


def discord_status_items() -> list[OpsStatusItem]:
    portfolio = os.getenv("OQP_DISCORD_WEBHOOK_URL") or os.getenv("OQP_HEALTH_WEBHOOK_URL")
    paper = os.getenv("OQP_PAPER_DISCORD_WEBHOOK_URL") or portfolio
    return [
        _webhook_status("Portfolio Discord webhook", portfolio),
        _webhook_status("Paper Discord webhook", paper),
    ]


def snapshot_notification_items() -> list[OpsStatusItem]:
    return [
        OpsStatusItem(
            "Notifications",
            "Server-side webhooks",
            "pass",
            "Local dashboard does not require webhook secrets; notification status comes from server jobs.",
        )
    ]


def safety_status_items(
    settings: OQPSettings,
    *,
    snapshot_mode: bool = False,
) -> list[OpsStatusItem]:
    items = [
        OpsStatusItem(
            "Safety",
            "Live trading disabled",
            "pass" if not settings.allow_live_trading else "fail",
            f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
        ),
        OpsStatusItem(
            "Safety",
            "Paper trading armed",
            "warn" if settings.allow_paper_trading else "pass",
            f"ALLOW_PAPER_TRADING={str(settings.allow_paper_trading).lower()}",
        ),
        OpsStatusItem(
            "Safety",
            "Paper order submit disabled",
            "pass" if not settings.allow_paper_order_submit else "warn",
            (
                "ALLOW_PAPER_ORDER_SUBMIT="
                f"{str(settings.allow_paper_order_submit).lower()}"
            ),
        ),
    ]
    if snapshot_mode:
        items.append(
            OpsStatusItem(
                "Safety",
                "Live monitor evidence",
                "pass",
                "Using synced server heartbeat and account snapshots for live monitoring.",
            )
        )
    else:
        items.append(
            OpsStatusItem(
                "Safety",
                "Live monitor gate",
                "pass" if settings.ibkr_live_monitor_enabled else "warn",
                f"IBKR_LIVE_MONITOR_ENABLED={str(settings.ibkr_live_monitor_enabled).lower()}",
            )
        )
    return items


def host_summary_values(repo_root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(repo_root)
    memory = _memory_summary()
    return {
        "disk_total_gb": disk.total / 1024**3,
        "disk_used_gb": disk.used / 1024**3,
        "disk_free_gb": disk.free / 1024**3,
        "disk_used_pct": disk.used / disk.total if disk.total else 0.0,
        **memory,
    }


def host_health_items(
    summary: dict[str, Any],
    *,
    snapshot_mode: bool = False,
) -> list[OpsStatusItem]:
    disk_used_pct = float(summary.get("disk_used_pct", 0.0))
    items = [
        OpsStatusItem(
            "Host",
            "Disk",
            "pass" if disk_used_pct < 0.85 else "warn",
            (
                f"used={disk_used_pct * 100:.1f}% "
                f"free={float(summary.get('disk_free_gb', 0.0)):.1f}GB"
            ),
        )
    ]
    memory_used_pct = summary.get("memory_used_pct")
    if memory_used_pct is None:
        items.append(
            OpsStatusItem(
                "Host",
                "Memory",
                "pass" if snapshot_mode else "warn",
                (
                    "Memory metrics are not collected from the local snapshot viewer."
                    if snapshot_mode
                    else "Memory metrics unavailable on this host."
                ),
            )
        )
    else:
        used = float(memory_used_pct)
        items.append(
            OpsStatusItem(
                "Host",
                "Memory",
                "pass" if used < 0.90 else "warn",
                f"used={used * 100:.1f}% free={float(summary.get('memory_free_gb', 0.0)):.1f}GB",
            )
        )
    return items


def command_status(
    cmd: list[str],
    *,
    timeout: float = 5.0,
    allow_codes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "fail", "detail": str(exc), "stdout": "", "stderr": ""}

    ok = completed.returncode in allow_codes
    stderr = completed.stderr.strip()
    detail = "ok" if ok else f"exit={completed.returncode} {stderr}".strip()
    return {
        "status": "pass" if ok else "fail",
        "detail": detail,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


def _webhook_status(name: str, url: str | None) -> OpsStatusItem:
    if not url:
        return OpsStatusItem("Notifications", name, "warn", "Webhook URL is not configured.")
    prefix_ok = url.startswith(
        ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/")
    )
    return OpsStatusItem(
        "Notifications",
        name,
        "pass" if prefix_ok else "warn",
        "Configured." if prefix_ok else "Configured value does not look like a Discord webhook URL.",
    )


def _systemd_props(output: str) -> dict[str, str]:
    props: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        props[key] = value
    return props


def _memory_summary() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return {"memory_used_pct": None, "memory_free_gb": None, "memory_total_gb": None}
    values: dict[str, float] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            values[parts[0].rstrip(":")] = float(parts[1]) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return {"memory_used_pct": None, "memory_free_gb": None, "memory_total_gb": None}
    used_pct = 1.0 - (available / total)
    return {
        "memory_used_pct": used_pct,
        "memory_free_gb": available / 1024**3,
        "memory_total_gb": total / 1024**3,
    }


def _file_age_hours(path: Path) -> float:
    modified = _mtime(path)
    return (datetime.now(timezone.utc) - modified).total_seconds() / 3600


def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _datetime_age_hours(value: Any) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600


def _float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value: Any) -> str:
    parsed = _float(value)
    return "n/a" if parsed is None else f"{parsed:,.2f}"


def _redact_account(account_id: Any) -> str:
    if account_id in (None, ""):
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
