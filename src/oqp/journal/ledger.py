"""SQLite ledger for operator journal notes and report references."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_JOURNAL_DB_PATH = REPO_ROOT / "runtime" / "db" / "journal" / "journal.db"


JOURNAL_ENTRIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS journal_entries (
    entry_id TEXT PRIMARY KEY,
    entry_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    category TEXT NOT NULL,
    environment TEXT,
    title TEXT NOT NULL,
    body TEXT,
    symbols_json TEXT NOT NULL,
    strategies_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    mistake TEXT,
    lesson TEXT,
    follow_up TEXT,
    metadata_json TEXT NOT NULL
)
"""


JOURNAL_ENTRY_INDEXES = (
    """
    CREATE INDEX IF NOT EXISTS idx_journal_entries_date
    ON journal_entries(entry_date DESC, created_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_journal_entries_category
    ON journal_entries(category, entry_date DESC)
    """,
)


@dataclass(frozen=True, slots=True)
class JournalEntryWriteResult:
    db_path: Path
    entry_id: str
    entry_date: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "entry_id": self.entry_id,
            "entry_date": self.entry_date,
            "category": self.category,
        }


def default_journal_ledger_path() -> Path:
    return DEFAULT_JOURNAL_DB_PATH


def ensure_journal_schema(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(JOURNAL_ENTRIES_SCHEMA)
        for statement in JOURNAL_ENTRY_INDEXES:
            conn.execute(statement)
        conn.commit()
    return path


def write_journal_entry(
    db_path: str | Path,
    *,
    entry_date: str | date | datetime | None = None,
    category: str,
    title: str,
    body: str | None = None,
    environment: str | None = None,
    symbols: tuple[str, ...] | list[str] = (),
    strategies: tuple[str, ...] | list[str] = (),
    tags: tuple[str, ...] | list[str] = (),
    mistake: str | None = None,
    lesson: str | None = None,
    follow_up: str | None = None,
    metadata: dict[str, Any] | None = None,
    entry_id: str | None = None,
    created_at: str | datetime | None = None,
) -> JournalEntryWriteResult:
    """Append one operator journal entry."""

    clean_category = _required_text(category, "category")
    clean_title = _required_text(title, "title")
    timestamp = _datetime_text(created_at or datetime.now(timezone.utc))
    date_value = _date_text(entry_date or datetime.now(timezone.utc))
    row_id = entry_id or f"journal-{uuid4().hex[:12]}"
    path = ensure_journal_schema(db_path)

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT INTO journal_entries (
                entry_id,
                entry_date,
                created_at,
                updated_at,
                category,
                environment,
                title,
                body,
                symbols_json,
                strategies_json,
                tags_json,
                mistake,
                lesson,
                follow_up,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                date_value,
                timestamp,
                timestamp,
                clean_category,
                _optional_text(environment),
                clean_title,
                _optional_text(body),
                json.dumps(_clean_items(symbols, uppercase=True), sort_keys=True),
                json.dumps(_clean_items(strategies), sort_keys=True),
                json.dumps(_clean_items(tags), sort_keys=True),
                _optional_text(mistake),
                _optional_text(lesson),
                _optional_text(follow_up),
                json.dumps(dict(metadata or {}), sort_keys=True),
            ),
        )
        conn.commit()

    return JournalEntryWriteResult(
        db_path=path,
        entry_id=row_id,
        entry_date=date_value,
        category=clean_category,
    )


def load_journal_entries(
    db_path: str | Path,
    *,
    category: str | None = None,
    entry_date: str | date | datetime | None = None,
    since: str | date | datetime | None = None,
    limit: int = 250,
) -> pd.DataFrame:
    columns = [
        "entry_id",
        "entry_date",
        "created_at",
        "updated_at",
        "category",
        "environment",
        "title",
        "body",
        "symbols_json",
        "strategies_json",
        "tags_json",
        "mistake",
        "lesson",
        "follow_up",
        "metadata_json",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    ensure_journal_schema(path)

    clauses: list[str] = []
    params: list[Any] = []
    if category:
        clauses.append("category = ?")
        params.append(str(category))
    if entry_date is not None:
        clauses.append("entry_date = ?")
        params.append(_date_text(entry_date))
    if since is not None:
        clauses.append("entry_date >= ?")
        params.append(_date_text(since))
    where = "WHERE " + " AND ".join(clauses) if clauses else ""

    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            f"""
            SELECT {", ".join(columns)}
            FROM journal_entries
            {where}
            ORDER BY entry_date DESC, created_at DESC
            LIMIT ?
            """,
            conn,
            params=(*params, max(int(limit), 1)),
        )


def _required_text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _datetime_text(value: str | datetime) -> str:
    if isinstance(value, datetime):
        active = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return active.replace(microsecond=0).isoformat()
    return str(value)


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def _clean_items(
    values: tuple[str, ...] | list[str],
    *,
    uppercase: bool = False,
) -> list[str]:
    cleaned = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        cleaned.append(text.upper() if uppercase else text)
    return sorted(set(cleaned))
