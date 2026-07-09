"""Market event calendar helpers for dashboard context."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_MARKET_EVENTS_PATH = REPO_ROOT / "runtime" / "state" / "market" / "events_calendar.json"
MARKET_EVENT_COLUMNS = ["Date", "Market", "Category", "Event", "Symbols", "Status", "Source", "Notes"]


def load_market_events(
    path: str | Path = DEFAULT_MARKET_EVENTS_PATH,
    *,
    start: date | datetime | str | None = None,
    horizon_days: int = 45,
) -> pd.DataFrame:
    """Load manually curated or provider-synced market events.

    This intentionally treats the local file/provider as the source of truth.
    LLMs can summarize events later, but should not invent exact calendar dates.
    """

    db_path = Path(path)
    if not db_path.exists():
        return pd.DataFrame(columns=MARKET_EVENT_COLUMNS)

    try:
        raw = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame(columns=MARKET_EVENT_COLUMNS)

    if isinstance(raw, dict):
        rows = raw.get("events", [])
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    normalized = [_normalize_event_row(row) for row in rows if isinstance(row, dict)]
    frame = pd.DataFrame(normalized, columns=MARKET_EVENT_COLUMNS)
    if frame.empty:
        return frame

    start_ts = pd.to_datetime(start or date.today(), errors="coerce")
    end_ts = start_ts + pd.Timedelta(days=int(horizon_days))
    event_dates = pd.to_datetime(frame["Date"], errors="coerce")
    in_window = event_dates.notna() & event_dates.ge(start_ts.normalize()) & event_dates.le(end_ts.normalize())
    out = frame.loc[in_window].copy()
    out["_date"] = event_dates.loc[in_window]
    return out.sort_values(["_date", "Market", "Category"]).drop(columns=["_date"]).reset_index(drop=True)


def seed_market_events_template(
    watchlist_symbols: Iterable[str],
    path: str | Path = DEFAULT_MARKET_EVENTS_PATH,
) -> Path:
    """Create a local event-calendar template if one does not exist."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        return db_path

    payload = {
        "schema": "oqp.market.events.v1",
        "notes": "Provider-synced or manually curated market events. Dates here are source-of-truth inputs for dashboards.",
        "watchlist_symbols": [str(symbol).upper().strip() for symbol in watchlist_symbols if str(symbol).strip()],
        "events": [],
    }
    db_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return db_path


def event_provider_plan(watchlist_symbols: Iterable[str]) -> pd.DataFrame:
    """Describe planned event-calendar feeds and the role of LLM commentary."""

    symbols = [str(symbol).upper().strip() for symbol in watchlist_symbols if str(symbol).strip()]
    symbol_preview = ", ".join(symbols[:12])
    if len(symbols) > 12:
        symbol_preview += f", +{len(symbols) - 12} more"

    rows = [
        {
            "Lane": "US macro",
            "Events": "FOMC, CPI, NFP, Treasury auctions, Fed speakers",
            "Primary Source": "official calendars / FMP economic calendar",
            "GLM Role": "summarize why the event matters and affected assets",
            "Status": "planned",
        },
        {
            "Lane": "Watchlist earnings",
            "Events": symbol_preview or "watchlist symbols",
            "Primary Source": "FMP earnings calendar or earnings-call transcript API",
            "GLM Role": "summarize call tone after transcripts arrive",
            "Status": "planned",
        },
        {
            "Lane": "China / HK",
            "Events": "PBoC, NBS macro releases, exchange holidays, futures delivery dates",
            "Primary Source": "Wind / QMT / broker calendar",
            "GLM Role": "translate and summarize Chinese event context",
            "Status": "planned",
        },
        {
            "Lane": "Global risk",
            "Events": "ECB, BOJ, Korea, Germany, commodities, crypto catalysts",
            "Primary Source": "provider calendar plus local manual overrides",
            "GLM Role": "produce morning briefing bullets",
            "Status": "planned",
        },
    ]
    return pd.DataFrame(rows)


def _normalize_event_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "Date": str(row.get("date") or row.get("Date") or "").strip(),
        "Market": str(row.get("market") or row.get("Market") or "").strip(),
        "Category": str(row.get("category") or row.get("Category") or "").strip(),
        "Event": str(row.get("event") or row.get("Event") or "").strip(),
        "Symbols": _string_list(row.get("symbols") or row.get("Symbols")),
        "Status": str(row.get("status") or row.get("Status") or "planned").strip(),
        "Source": str(row.get("source") or row.get("Source") or "local").strip(),
        "Notes": str(row.get("notes") or row.get("Notes") or "").strip(),
    }


def _string_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item).upper().strip() for item in value if str(item).strip())
    return str(value or "").strip()
