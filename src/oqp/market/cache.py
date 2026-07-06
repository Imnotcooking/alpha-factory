"""Persistent market data cache for dashboard reads."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_MARKET_CACHE_PATH = REPO_ROOT / "runtime" / "db" / "market" / "market_cache.db"
DEFAULT_MARKET_CACHE_MAX_AGE_HOURS = 18.0

EXPECTED_HISTORY_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def ensure_market_cache_schema(path: str | Path = DEFAULT_MARKET_CACHE_PATH) -> Path:
    """Create the market cache schema if it does not already exist."""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_history (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                source TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (symbol, date, source)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_market_history_symbol_date
            ON market_history (symbol, date)
            """
        )
        conn.commit()
    return db_path


def load_cached_market_history(
    symbol: str,
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    source: str = "yahoo",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Load yfinance-shaped OHLCV history for one symbol from SQLite."""

    symbol_key = normalize_symbol(symbol)
    if not symbol_key:
        return empty_history_frame()

    db_path = ensure_market_cache_schema(path)
    params: list[Any] = [symbol_key, source]
    where = "symbol = ? AND source = ?"
    if lookback_days is not None:
        start = (datetime.now(timezone.utc).date() - timedelta(days=int(lookback_days))).isoformat()
        where += " AND date >= ?"
        params.append(start)

    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT date, open, high, low, close, adj_close, volume
            FROM market_history
            WHERE {where}
            ORDER BY date
            """,
            conn,
            params=params,
        )
    if frame.empty:
        return empty_history_frame()

    frame["Date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date")
    out = pd.DataFrame(index=frame.index)
    out["Open"] = pd.to_numeric(frame["open"], errors="coerce")
    out["High"] = pd.to_numeric(frame["high"], errors="coerce")
    out["Low"] = pd.to_numeric(frame["low"], errors="coerce")
    out["Close"] = pd.to_numeric(frame["close"], errors="coerce")
    out["Adj Close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    out["Volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    return out.dropna(subset=["Close"]).sort_index()


def load_cached_price_history(
    symbols: Iterable[str],
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    source: str = "yahoo",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Load cached OHLCV rows as long-form symbol/date/close history."""

    requested = [symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol]
    requested = list(dict.fromkeys(requested))
    columns = ["symbol", "date", "close"]
    if not requested:
        return pd.DataFrame(columns=columns)

    rows = []
    for symbol in requested:
        history = load_cached_market_history(
            symbol,
            path=path,
            source=source,
            lookback_days=lookback_days,
        )
        if history.empty:
            continue
        frame = history.reset_index().rename(columns={"Date": "date", "Close": "close"})
        frame["symbol"] = symbol
        rows.append(frame[columns])

    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    return out.dropna(subset=["date", "close"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def write_market_history(
    symbol: str,
    history: pd.DataFrame,
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    source: str = "yahoo",
    fetched_at: datetime | None = None,
) -> int:
    """Persist OHLCV rows for one symbol and return rows written."""

    symbol_key = normalize_symbol(symbol)
    if not symbol_key:
        return 0

    normalized = normalize_history_frame(history)
    if normalized.empty:
        return 0

    fetched_text = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )
    rows = [
        (
            symbol_key,
            row["date"],
            source,
            _optional_float(row["open"]),
            _optional_float(row["high"]),
            _optional_float(row["low"]),
            _optional_float(row["close"]),
            _optional_float(row["adj_close"]),
            _optional_float(row["volume"]),
            fetched_text,
        )
        for row in normalized.to_dict("records")
    ]

    db_path = ensure_market_cache_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            """
            INSERT INTO market_history (
                symbol, date, source, open, high, low, close, adj_close, volume, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date, source) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                adj_close = excluded.adj_close,
                volume = excluded.volume,
                fetched_at = excluded.fetched_at
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def market_cache_status(
    symbols: Iterable[str],
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    source: str = "yahoo",
    max_age_hours: float = DEFAULT_MARKET_CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Return one cache-health row per requested symbol."""

    requested = [symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol]
    requested = list(dict.fromkeys(requested))
    columns = ["Symbol", "Rows", "First Date", "Last Date", "Fetched At", "Age Hours", "State"]
    if not requested:
        return pd.DataFrame(columns=columns)

    db_path = ensure_market_cache_schema(path)
    placeholders = ",".join("?" for _ in requested)
    params = [*requested, source]
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT symbol,
                   COUNT(*) AS rows,
                   MIN(date) AS first_date,
                   MAX(date) AS last_date,
                   MAX(fetched_at) AS fetched_at
            FROM market_history
            WHERE symbol IN ({placeholders})
              AND source = ?
            GROUP BY symbol
            """,
            conn,
            params=params,
        )

    lookup = {
        str(row["symbol"]).upper(): row
        for row in frame.to_dict("records")
    }
    now = datetime.now(timezone.utc)
    rows = []
    for symbol in requested:
        row = lookup.get(symbol)
        if not row:
            rows.append(
                {
                    "Symbol": symbol,
                    "Rows": 0,
                    "First Date": "",
                    "Last Date": "",
                    "Fetched At": "",
                    "Age Hours": None,
                    "State": "missing",
                }
            )
            continue

        age = cache_age_hours(row.get("fetched_at"), now=now)
        state = "stale" if age is None or age > max_age_hours else "fresh"
        rows.append(
            {
                "Symbol": symbol,
                "Rows": int(row.get("rows") or 0),
                "First Date": row.get("first_date") or "",
                "Last Date": row.get("last_date") or "",
                "Fetched At": row.get("fetched_at") or "",
                "Age Hours": age,
                "State": state,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def refresh_yahoo_market_cache(
    symbols: Iterable[str],
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    period: str = "1y",
    provider: Callable[[str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fetch Yahoo history for symbols and persist it to the market cache."""

    requested = [symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol]
    requested = list(dict.fromkeys(requested))
    if provider is None:
        provider = fetch_yahoo_history

    rows = []
    for symbol in requested:
        try:
            history = provider(symbol, period)
            written = write_market_history(symbol, history, path=path, source="yahoo")
        except Exception as exc:  # pragma: no cover - defensive for vendor/network failures.
            rows.append(
                {
                    "Symbol": symbol,
                    "Status": "error",
                    "Rows": 0,
                    "Detail": str(exc),
                }
            )
            continue

        rows.append(
            {
                "Symbol": symbol,
                "Status": "ok" if written else "empty",
                "Rows": written,
                "Detail": "" if written else "provider returned no usable close history",
            }
        )
    return pd.DataFrame(rows, columns=["Symbol", "Status", "Rows", "Detail"])


def fetch_yahoo_history(symbol: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV history from yfinance."""

    import yfinance as yf

    history = yf.Ticker(symbol).history(period=period)
    return history if isinstance(history, pd.DataFrame) else pd.DataFrame()


def normalize_history_frame(history: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance-like OHLCV history into cache columns."""

    if history is None or history.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    frame = history.copy()
    if "Date" not in frame.columns:
        frame = frame.reset_index()

    columns = {str(column).lower().replace("_", " ").strip(): column for column in frame.columns}
    date_col = _first_existing(columns, ("date", "datetime", "timestamp", "index"))
    close_col = _first_existing(columns, ("close", "adj close", "adj_close", "price"))
    if date_col is None or close_col is None:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype("string")
    out["open"] = _numeric_column(frame, columns, ("open",))
    out["high"] = _numeric_column(frame, columns, ("high",))
    out["low"] = _numeric_column(frame, columns, ("low",))
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce")
    out["adj_close"] = _numeric_column(frame, columns, ("adj close", "adj_close", "close"))
    out["volume"] = _numeric_column(frame, columns, ("volume",))
    out = out.dropna(subset=["date", "close"])
    out = out[out["date"].astype(str).ne("")]
    return out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)


def empty_history_frame() -> pd.DataFrame:
    """Return an empty yfinance-shaped history frame."""

    return pd.DataFrame(columns=EXPECTED_HISTORY_COLUMNS)


def cache_age_hours(value: Any, *, now: datetime | None = None) -> float | None:
    """Compute cache age in hours from an ISO timestamp."""

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return round((current.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 3600, 2)


def normalize_symbol(value: object) -> str:
    """Normalize market symbols for cache keys."""

    return str(value or "").strip().upper()


def _numeric_column(
    frame: pd.DataFrame,
    columns: dict[str, object],
    candidates: tuple[str, ...],
) -> pd.Series:
    column = _first_existing(columns, candidates)
    if column is None:
        return pd.Series([None] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _first_existing(columns: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed
