from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


SEED_SCHEMA_VERSION = "tick_pulse_hypothesis_seed_v1"


def make_hypothesis_seed_id(payload: dict[str, Any]) -> str:
    core = {
        "schema_version": SEED_SCHEMA_VERSION,
        "source_file": payload.get("source_file", ""),
        "symbol": payload.get("symbol", ""),
        "event_time": payload.get("event_time", ""),
        "pulse_direction": payload.get("pulse_direction", ""),
        "behavior": payload.get("behavior", ""),
        "rule": payload.get("rule", {}),
    }
    normalized = json.dumps(core, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def ensure_hypothesis_seed_table(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tick_pulse_hypothesis_seeds (
                seed_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                name TEXT NOT NULL,
                source_file TEXT NOT NULL,
                product TEXT,
                symbol TEXT NOT NULL,
                event_time TEXT,
                pulse_direction TEXT,
                behavior TEXT,
                quality_status TEXT,
                pulse_family TEXT,
                rule_json TEXT NOT NULL,
                example_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def save_hypothesis_seed(db_path: str, payload: dict[str, Any]) -> str:
    ensure_hypothesis_seed_table(db_path)
    seed_id = str(payload.get("seed_id") or make_hypothesis_seed_id(payload))
    rule = payload.get("rule", {})
    example = payload.get("example", {})

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tick_pulse_hypothesis_seeds (
                seed_id, schema_version, name, source_file, product, symbol,
                event_time, pulse_direction, behavior, quality_status,
                pulse_family, rule_json, example_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                seed_id,
                SEED_SCHEMA_VERSION,
                str(payload.get("name") or f"Pulse seed {seed_id}"),
                str(payload.get("source_file") or ""),
                str(payload.get("product") or ""),
                str(payload.get("symbol") or ""),
                str(payload.get("event_time") or ""),
                str(payload.get("pulse_direction") or ""),
                str(payload.get("behavior") or ""),
                str(payload.get("quality_status") or ""),
                str(payload.get("pulse_family") or ""),
                json.dumps(rule, sort_keys=True, ensure_ascii=False, default=str),
                json.dumps(example, sort_keys=True, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
    return seed_id


def list_hypothesis_seeds(db_path: str, limit: int = 100) -> list[dict[str, Any]]:
    ensure_hypothesis_seed_table(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM tick_pulse_hypothesis_seeds
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_decode_seed_row(dict(row)) for row in rows]


def get_hypothesis_seed(db_path: str, seed_id: str) -> dict[str, Any] | None:
    ensure_hypothesis_seed_table(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tick_pulse_hypothesis_seeds WHERE seed_id = ?",
            (str(seed_id),),
        ).fetchone()
    return _decode_seed_row(dict(row)) if row else None


def format_seed_label(seed: dict[str, Any]) -> str:
    time_text = str(seed.get("event_time") or "")
    if len(time_text) > 19:
        time_text = time_text[:19]
    return (
        f"{seed.get('name', 'Pulse seed')} | {seed.get('symbol', '')} | "
        f"{seed.get('behavior', '')} | {time_text}"
    )


def _decode_seed_row(row: dict[str, Any]) -> dict[str, Any]:
    row["rule"] = _loads(row.pop("rule_json", "{}"))
    row["example"] = _loads(row.pop("example_json", "{}"))
    return row


def _loads(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value or "{}")
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}
