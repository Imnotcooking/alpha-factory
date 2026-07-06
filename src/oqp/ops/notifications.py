"""Shared notification helpers for command entrypoints and jobs."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, TextIO


def env_file_value(path: str | Path, name: str) -> str | None:
    """Return a simple KEY=value entry from an env file without sourcing it."""

    env_path = Path(path)
    if not env_path.exists():
        return None
    prefix = f"{name}="
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith(prefix):
            continue
        value = stripped[len(prefix) :].strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        return value or None
    return None


def first_env_file_value(path: str | Path, names: Iterable[str]) -> str | None:
    for name in names:
        value = env_file_value(path, name)
        if value:
            return value
    return None


def discord_field(name: str, value: Any, *, inline: bool = False) -> dict[str, Any]:
    text = str(value).strip() or "No detail."
    return {
        "name": name[:256],
        "value": text[:1024],
        "inline": inline,
    }


def post_json_webhook(
    url: str,
    payload: dict[str, Any],
    *,
    user_agent: str,
    label: str = "webhook",
    timeout: float = 10,
    stream: TextIO | None = None,
) -> bool:
    """Post JSON to a webhook and report non-fatal delivery errors."""

    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        },
        method="POST",
    )
    target = stream or sys.stderr
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        print(
            f"WARN   {label}  HTTP {exc.code}: {detail or exc.reason}",
            file=target,
        )
        return False
    except (OSError, urllib.error.URLError) as exc:
        print(f"WARN   {label}  {exc}", file=target)
        return False
    return True
