#!/usr/bin/env python3
"""Create ~/.oqp_server_env from an already-running Alpha Factory server.

This migration helper reads secrets from the running IBKR Docker containers and
existing private env files, then writes a consolidated server env file without
printing secret values.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HOME = Path.home()
REPO = Path(os.environ.get("OQP_REPO_ROOT", "/home/ubuntu/oqp_new"))
TARGET = Path(os.environ.get("OQP_SERVER_ENV", HOME / ".oqp_server_env"))


def docker_env(name: str) -> tuple[str, dict[str, str]]:
    raw = subprocess.check_output(["sudo", "docker", "inspect", name], text=True)
    data = json.loads(raw)[0]
    image = data.get("Config", {}).get("Image") or "ghcr.io/gnzsnz/ib-gateway:latest"
    values: dict[str, str] = {}
    for item in data.get("Config", {}).get("Env") or []:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return image, values


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parts = shlex.split(value, posix=True)
            parsed = parts[0] if len(parts) == 1 else value
        except ValueError:
            parsed = value.strip("'\"")
        if key:
            values[key] = parsed
    return values


def quote(value: str) -> str:
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    return f'"{escaped}"'


def main() -> int:
    live_image, live_env = docker_env("ib-gateway-live")
    paper_image, paper_env = docker_env("ib-gateway-paper")
    portfolio_env = parse_env_file(HOME / ".oqp_portfolio_health_env")
    paper_notify_env = parse_env_file(HOME / ".oqp_paper_trading_env")
    repo_env = parse_env_file(REPO / ".env")

    values = {
        "OQP_REPO_ROOT": str(REPO),
        "PYTHONPATH": "src:.",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "IB_GATEWAY_IMAGE": live_image or paper_image or "ghcr.io/gnzsnz/ib-gateway:latest",
        "IBKR_LIVE_USER": live_env.get("TWS_USERID", ""),
        "IBKR_LIVE_PASSWORD": live_env.get("TWS_PASSWORD", ""),
        "IBKR_PAPER_USER": paper_env.get("TWS_USERID", ""),
        "IBKR_PAPER_PASSWORD": paper_env.get("TWS_PASSWORD", ""),
        "IBKR_VNC_PASSWORD": live_env.get("VNC_SERVER_PASSWORD")
        or paper_env.get("VNC_SERVER_PASSWORD", "")
        or "REPLACE_ME_BEFORE_CONTAINER_RECREATE",
        "IBKR_LIVE_API_PORT": "4001",
        "IBKR_LIVE_CONTAINER_API_PORT": "4001",
        "IBKR_LIVE_VNC_PORT": "5901",
        "IBKR_PAPER_API_PORT": "7497",
        "IBKR_PAPER_CONTAINER_API_PORT": "4004",
        "IBKR_PAPER_VNC_PORT": "5902",
        "IBKR_EXISTING_SESSION_ACTION": live_env.get("EXISTING_SESSION_DETECTED_ACTION")
        or paper_env.get("EXISTING_SESSION_DETECTED_ACTION")
        or "primary",
        "IBKR_PAPER_READ_ONLY_API": paper_env.get("READ_ONLY_API", "yes"),
        "IBKR_HOST": "127.0.0.1",
        "IBKR_LIVE_PORT": "4001",
        "IBKR_PAPER_PORT": "7497",
        "IBKR_LIVE_CLIENT_ID": repo_env.get("IBKR_LIVE_CLIENT_ID", "201"),
        "IBKR_PAPER_CLIENT_ID": repo_env.get("IBKR_PAPER_CLIENT_ID", "101"),
        "IBKR_LIVE_MONITOR_ENABLED": "true",
        "ALLOW_LIVE_TRADING": "false",
        "ALLOW_PAPER_TRADING": repo_env.get("ALLOW_PAPER_TRADING", "false"),
        "PAPER_MAX_ORDER_NOTIONAL": repo_env.get("PAPER_MAX_ORDER_NOTIONAL", "10000"),
        "PAPER_MAX_DAILY_NOTIONAL": repo_env.get("PAPER_MAX_DAILY_NOTIONAL", "50000"),
        "PAPER_ALLOWED_ASSET_CLASSES": repo_env.get(
            "PAPER_ALLOWED_ASSET_CLASSES", "equity,etf"
        ),
        "PAPER_OPTIONS_ENABLED": repo_env.get("PAPER_OPTIONS_ENABLED", "false"),
        "FMP_API_KEY": repo_env.get("FMP_API_KEY", ""),
        "MASSIVE_API_KEY": repo_env.get("MASSIVE_API_KEY", ""),
        "POLYGON_API_KEY": repo_env.get("POLYGON_API_KEY", ""),
        "OPTIONS_API_KEY": repo_env.get("OPTIONS_API_KEY", ""),
        "OQP_DISCORD_WEBHOOK_URL": portfolio_env.get(
            "OQP_DISCORD_WEBHOOK_URL", repo_env.get("OQP_DISCORD_WEBHOOK_URL", "")
        ),
        "OQP_PAPER_DISCORD_WEBHOOK_URL": paper_notify_env.get(
            "OQP_PAPER_DISCORD_WEBHOOK_URL",
            repo_env.get("OQP_PAPER_DISCORD_WEBHOOK_URL", ""),
        ),
        "OQP_PORTFOLIO_HEALTH_MAX_AGE_HOURS": portfolio_env.get(
            "OQP_PORTFOLIO_HEALTH_MAX_AGE_HOURS", "36"
        ),
        "OQP_PAPER_HEALTH_MAX_AGE_HOURS": paper_notify_env.get(
            "OQP_PAPER_HEALTH_MAX_AGE_HOURS", "36"
        ),
    }

    required = (
        "IBKR_LIVE_USER",
        "IBKR_LIVE_PASSWORD",
        "IBKR_PAPER_USER",
        "IBKR_PAPER_PASSWORD",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        print("Missing required values: " + ", ".join(missing))
        return 1

    if TARGET.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = TARGET.with_name(f"{TARGET.name}.backup.{stamp}")
        TARGET.replace(backup)
        os.chmod(backup, 0o600)

    lines = [
        "# Alpha Factory server env. Generated from existing server state.",
        "# Contains secrets. Do not copy into git.",
    ]
    lines.extend(f"{key}={quote(value)}" for key, value in values.items())

    tmp = TARGET.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(TARGET)
    os.chmod(TARGET, 0o600)

    report = {
        "path": str(TARGET),
        "mode": oct(TARGET.stat().st_mode & 0o777),
        "keys_written": len(values),
        "discord_configured": bool(values.get("OQP_DISCORD_WEBHOOK_URL")),
        "paper_discord_configured": bool(values.get("OQP_PAPER_DISCORD_WEBHOOK_URL")),
        "vendor_keys_present": sorted(
            key
            for key in ("FMP_API_KEY", "MASSIVE_API_KEY", "POLYGON_API_KEY", "OPTIONS_API_KEY")
            if values.get(key)
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
