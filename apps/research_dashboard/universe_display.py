from __future__ import annotations

import math
from typing import Any


_ALL_MARKERS = {"", "ALL", "NONE", "NAN", "<NA>", "NULL"}


def parse_traded_tickers(value: Any) -> list[str]:
    """Parse a stored comma-delimited ticker scope into stable unique tokens."""

    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    raw = str(value).strip()
    if raw.upper() in _ALL_MARKERS:
        return []

    tickers: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        ticker = item.strip()
        if not ticker or ticker.upper() in _ALL_MARKERS or ticker in seen:
            continue
        tickers.append(ticker)
        seen.add(ticker)
    return tickers


def coerce_universe_size(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        size = int(float(value))
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def summarize_traded_universe(
    traded_tickers: Any,
    universe_size: Any = None,
    *,
    max_tickers: int = 10,
) -> str:
    """Return a compact display label for a run's traded universe."""

    tickers = parse_traded_tickers(traded_tickers)
    size = coerce_universe_size(universe_size)
    if not tickers or (size is not None and len(tickers) >= size):
        return "ALL"

    if max_tickers <= 0:
        return f"{len(tickers):,} selected"
    if len(tickers) <= max_tickers:
        return ", ".join(tickers)
    visible = ", ".join(tickers[:max_tickers])
    return f"{len(tickers):,} selected: {visible}, ..."


def traded_universe_detail(traded_tickers: Any, universe_size: Any = None) -> str:
    summary = summarize_traded_universe(traded_tickers, universe_size, max_tickers=10)
    size = coerce_universe_size(universe_size)
    if summary == "ALL":
        return f"ALL ({size:,} assets)" if size else "ALL"
    return f"{summary} (Total Pool: {size:,})" if size else summary
