"""SQLite persistence for discretionary opportunity hub snapshots."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_OPPORTUNITY_HISTORY_DB_PATH = REPO_ROOT / "runtime" / "db" / "investing" / "opportunity_history.db"


def ensure_opportunity_history_schema(path: str | Path = DEFAULT_OPPORTUNITY_HISTORY_DB_PATH) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS opportunity_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                spot REAL,
                action_bucket TEXT,
                primary_route TEXT,
                direction TEXT,
                direction_score REAL,
                news_tone TEXT,
                news_score REAL,
                target_upside REAL,
                forecast_vol REAL,
                market_iv REAL,
                reference_expiry TEXT,
                lens_json TEXT NOT NULL,
                route_json TEXT NOT NULL,
                playbook_json TEXT NOT NULL,
                catalyst_json TEXT NOT NULL,
                thesis_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_opportunity_snapshots_symbol_captured
            ON opportunity_snapshots (symbol, captured_at DESC)
            """
        )
        conn.commit()
    return db_path


def write_opportunity_snapshot(
    *,
    symbol: str,
    spot: float,
    action_bucket: str,
    primary_route: str,
    direction: str,
    direction_score: float,
    news_tone: str,
    news_score: float,
    target_upside: float | None,
    forecast_vol: float,
    market_iv: float,
    reference_expiry: str,
    lens_frame: pd.DataFrame,
    route_frame: pd.DataFrame,
    playbook_frame: pd.DataFrame,
    catalyst_frame: pd.DataFrame,
    thesis_metadata: dict[str, Any] | None = None,
    path: str | Path = DEFAULT_OPPORTUNITY_HISTORY_DB_PATH,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Persist one hub snapshot and return summary metadata."""

    symbol_key = normalize_symbol(symbol)
    captured = (captured_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    captured_text = captured.isoformat(timespec="seconds")
    snapshot_id = f"opp-{symbol_key}-{captured.strftime('%Y%m%dT%H%M%SZ')}"
    db_path = ensure_opportunity_history_schema(path)
    row = (
        snapshot_id,
        symbol_key,
        captured_text,
        safe_float(spot),
        str(action_bucket or ""),
        str(primary_route or ""),
        str(direction or ""),
        safe_float(direction_score),
        str(news_tone or ""),
        safe_float(news_score),
        safe_float(target_upside),
        safe_float(forecast_vol),
        safe_float(market_iv),
        str(reference_expiry or ""),
        frame_to_json(lens_frame),
        frame_to_json(route_frame),
        frame_to_json(playbook_frame),
        frame_to_json(catalyst_frame),
        json.dumps(thesis_metadata or {}, default=str, sort_keys=True),
    )
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO opportunity_snapshots (
                snapshot_id, symbol, captured_at, spot, action_bucket, primary_route,
                direction, direction_score, news_tone, news_score, target_upside,
                forecast_vol, market_iv, reference_expiry, lens_json, route_json,
                playbook_json, catalyst_json, thesis_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        conn.commit()
    return {
        "snapshot_id": snapshot_id,
        "symbol": symbol_key,
        "captured_at": captured_text,
        "db_path": str(db_path),
    }


def load_opportunity_history(
    symbol: str | None = None,
    *,
    path: str | Path = DEFAULT_OPPORTUNITY_HISTORY_DB_PATH,
    limit: int = 50,
) -> pd.DataFrame:
    """Load compact opportunity snapshot history for dashboard display."""

    db_path = ensure_opportunity_history_schema(path)
    query = """
        SELECT snapshot_id, symbol, captured_at, spot, action_bucket, primary_route,
               direction, direction_score, news_tone, news_score, target_upside,
               forecast_vol, market_iv, reference_expiry
        FROM opportunity_snapshots
    """
    params: tuple[Any, ...] = ()
    symbol_key = normalize_symbol(symbol) if symbol else ""
    if symbol_key:
        query += " WHERE symbol = ?"
        params = (symbol_key,)
    query += " ORDER BY captured_at DESC LIMIT ?"
    params = (*params, int(limit))
    with closing(sqlite3.connect(db_path)) as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    return frame if not frame.empty else empty_history_frame()


def load_opportunity_snapshot_detail(
    snapshot_id: str,
    *,
    path: str | Path = DEFAULT_OPPORTUNITY_HISTORY_DB_PATH,
) -> dict[str, Any]:
    db_path = ensure_opportunity_history_schema(path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            """
            SELECT lens_json, route_json, playbook_json, catalyst_json, thesis_json
            FROM opportunity_snapshots
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
    if row is None:
        return {}
    return {
        "lens": json.loads(row[0] or "[]"),
        "route": json.loads(row[1] or "[]"),
        "playbook": json.loads(row[2] or "[]"),
        "catalyst": json.loads(row[3] or "[]"),
        "thesis": json.loads(row[4] or "{}"),
    }


def frame_to_json(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "[]"
    return json.dumps(frame.to_dict("records"), default=str, sort_keys=True)


def normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "snapshot_id",
            "symbol",
            "captured_at",
            "spot",
            "action_bucket",
            "primary_route",
            "direction",
            "direction_score",
            "news_tone",
            "news_score",
            "target_upside",
            "forecast_vol",
            "market_iv",
            "reference_expiry",
        ]
    )
