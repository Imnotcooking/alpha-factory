#!/usr/bin/env python3
"""Check IBKR API handshakes and optionally post a Discord heartbeat report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.config import load_settings  # noqa: E402
from oqp.ops import (  # noqa: E402
    DEFAULT_IBKR_HEARTBEAT_HEALTH_PATH,
    OpsStatusItem,
    ibkr_api_heartbeat_item,
)


PROFILE_SPECS = {
    "live": ("Live IBKR API heartbeat", "ibkr_live_readonly"),
    "paper": ("Paper IBKR API heartbeat", "ibkr_paper_readonly"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check read-only IBKR API heartbeat status.",
    )
    parser.add_argument(
        "--profile",
        choices=("all", "live", "paper"),
        default="all",
        help="Which IBKR profile to check.",
    )
    parser.add_argument(
        "--env-file",
        default=str(REPO_ROOT / ".env"),
        help="Path to the runtime .env file.",
    )
    parser.add_argument(
        "--status-path",
        default=str(DEFAULT_IBKR_HEARTBEAT_HEALTH_PATH),
        help="Where to write machine-readable heartbeat status JSON.",
    )
    parser.add_argument(
        "--webhook-url",
        default=(
            os.getenv("OQP_IBKR_HEARTBEAT_WEBHOOK_URL")
            or os.getenv("OQP_DISCORD_WEBHOOK_URL")
            or os.getenv("OQP_HEALTH_WEBHOOK_URL")
        ),
        help="Optional Discord webhook URL.",
    )
    parser.add_argument(
        "--notify-always",
        action="store_true",
        help="Post webhook status even when all checks pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.env_file)
    items = run_checks(settings, profile=args.profile)
    payload = _status_payload(items)

    status_path = Path(args.status_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print_checks(items)

    failed = any(item.status == "fail" for item in items)
    if args.webhook_url and (failed or args.notify_always):
        _post_webhook(args.webhook_url, payload)

    return 1 if failed else 0


def run_checks(settings: Any, *, profile: str) -> list[OpsStatusItem]:
    names = ("live", "paper") if profile == "all" else (profile,)
    return [
        ibkr_api_heartbeat_item(PROFILE_SPECS[name][0], PROFILE_SPECS[name][1], settings)
        for name in names
    ]


def print_checks(items: list[OpsStatusItem]) -> None:
    width = max(len(item.name) for item in items) if items else 0
    for item in items:
        print(f"{item.status.upper():5}  {item.name:<{width}}  {item.detail}")


def _status_payload(items: list[OpsStatusItem]) -> dict[str, Any]:
    status = "pass"
    if any(item.status == "fail" for item in items):
        status = "fail"
    elif any(item.status == "warn" for item in items):
        status = "warn"

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": [
            {
                "name": item.name,
                "status": item.status,
                "detail": item.detail,
                "metadata": item.metadata,
            }
            for item in items
        ],
    }


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    body = json.dumps(_discord_payload(payload), sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "OQP-IBKR-Heartbeat/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   webhook  Could not post IBKR heartbeat: "
            f"HTTP {exc.code}: {detail or exc.reason}",
            file=sys.stderr,
        )
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARN   webhook  Could not post IBKR heartbeat: {exc}", file=sys.stderr)


def _discord_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "unknown")).upper()
    checks = payload.get("checks", [])
    if not isinstance(checks, list):
        checks = []

    fields = []
    for check in checks[:8]:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name", "check"))
        detail = str(check.get("detail", ""))
        fields.append(
            {
                "name": f"{str(check.get('status', '')).upper()}: {name}"[:256],
                "value": detail[:1024] or "No detail.",
                "inline": False,
            }
        )

    return {
        "username": "OQP IBKR Heartbeat",
        "content": f"OQP IBKR API heartbeat: {status}",
        "allowed_mentions": {"parse": []},
        "embeds": [
            {
                "title": "IBKR API Heartbeat",
                "description": f"Status: {status}",
                "color": 0x2ECC71 if status == "PASS" else 0xE74C3C,
                "timestamp": payload.get("checked_at"),
                "fields": fields,
            }
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
