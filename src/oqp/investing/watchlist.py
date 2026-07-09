"""Runtime watchlist storage for the investing dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from oqp.config import REPO_ROOT

DEFAULT_INVESTING_STATE_DIR = REPO_ROOT / "runtime" / "state" / "investing"
DEFAULT_STOCK_WATCHLIST_PATH = DEFAULT_INVESTING_STATE_DIR / "stock_watchlist.json"


def normalize_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def normalize_watchlist(symbols: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_symbol in symbols:
        symbol = normalize_symbol(raw_symbol)
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def _read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if isinstance(payload, list):
        return normalize_watchlist(payload)
    if isinstance(payload, dict):
        raw_symbols = payload.get("symbols") or payload.get("watchlist") or []
        if isinstance(raw_symbols, list):
            return normalize_watchlist(raw_symbols)
    return []


def load_stock_watchlist(path: Path | None = None) -> list[str]:
    """Load the canonical runtime watchlist."""

    target = path or DEFAULT_STOCK_WATCHLIST_PATH
    return _read_watchlist(target)


def save_stock_watchlist(
    symbols: Iterable[object],
    path: Path | None = None,
) -> Path:
    """Persist symbols to the canonical runtime watchlist path."""

    target = path or DEFAULT_STOCK_WATCHLIST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_watchlist(symbols)
    target.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return target


def add_stock_watchlist_symbol(symbol: str, path: Path | None = None) -> list[str]:
    watchlist = load_stock_watchlist(path)
    symbol = normalize_symbol(symbol)
    if symbol and symbol not in watchlist:
        watchlist.append(symbol)
        save_stock_watchlist(watchlist, path)
    return watchlist


def remove_stock_watchlist_symbol(symbol: str, path: Path | None = None) -> list[str]:
    target_symbol = normalize_symbol(symbol)
    watchlist = [item for item in load_stock_watchlist(path) if item != target_symbol]
    save_stock_watchlist(watchlist, path)
    return watchlist
