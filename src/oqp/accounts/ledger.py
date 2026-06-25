"""SQLite storage for unified live, paper, and future account snapshots."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.accounts.models import AccountSnapshot
from oqp.config.paths import REPO_ROOT


DEFAULT_ACCOUNT_LEDGER_PATH = REPO_ROOT / "runtime" / "db" / "accounts" / "account_ledger.db"

ACCOUNT_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    account_key TEXT NOT NULL,
    as_of TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    environment TEXT NOT NULL,
    currency TEXT NOT NULL,
    net_liquidation REAL,
    cash REAL,
    buying_power REAL,
    gross_position_value REAL,
    margin_buffer REAL,
    position_count INTEGER NOT NULL,
    metadata_json TEXT
)
"""

ACCOUNT_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_positions (
    snapshot_id TEXT NOT NULL,
    account_key TEXT NOT NULL,
    as_of TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    environment TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_cost REAL,
    market_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    currency TEXT NOT NULL,
    multiplier REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT,
    PRIMARY KEY (snapshot_id, broker, symbol, asset_class)
)
"""

ACCOUNT_CASH_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_cash (
    snapshot_id TEXT NOT NULL,
    account_key TEXT NOT NULL,
    as_of TEXT NOT NULL,
    snapshot_date TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    environment TEXT NOT NULL,
    currency TEXT NOT NULL,
    cash REAL NOT NULL,
    settled_cash REAL,
    buying_power REAL,
    metadata_json TEXT,
    PRIMARY KEY (snapshot_id, currency)
)
"""

ACCOUNT_NAV_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_nav (
    date TEXT NOT NULL,
    account_key TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    environment TEXT NOT NULL,
    as_of TEXT NOT NULL,
    net_liquidation REAL NOT NULL,
    cash REAL,
    daily_pnl REAL,
    position_count INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL,
    PRIMARY KEY (date, account_key)
)
"""

ACCOUNT_TRADE_EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_trade_events (
    event_id TEXT PRIMARY KEY,
    account_key TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    environment TEXT NOT NULL,
    event_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT,
    quantity REAL,
    price REAL,
    commission REAL,
    currency TEXT,
    occurred_at TEXT NOT NULL,
    strategy_id TEXT,
    order_id TEXT,
    broker_order_id TEXT,
    metadata_json TEXT
)
"""


@dataclass(frozen=True, slots=True)
class AccountSnapshotWriteResult:
    db_path: Path
    snapshot_id: str
    account_key: str
    snapshot_date: str
    environment: str
    profile: str
    account_id: str | None
    position_rows: int
    cash_rows: int
    net_liquidation: float
    daily_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "snapshot_id": self.snapshot_id,
            "account_key": self.account_key,
            "snapshot_date": self.snapshot_date,
            "environment": self.environment,
            "profile": self.profile,
            "account_id": _redact_account(self.account_id),
            "position_rows": self.position_rows,
            "cash_rows": self.cash_rows,
            "net_liquidation": self.net_liquidation,
            "daily_pnl": self.daily_pnl,
        }


def default_account_ledger_path() -> Path:
    configured = os.getenv("OQP_ACCOUNT_LEDGER_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_ACCOUNT_LEDGER_PATH


def ensure_account_ledger_schema(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(ACCOUNT_SNAPSHOTS_SCHEMA)
        conn.execute(ACCOUNT_POSITIONS_SCHEMA)
        conn.execute(ACCOUNT_CASH_SCHEMA)
        conn.execute(ACCOUNT_NAV_SCHEMA)
        conn.execute(ACCOUNT_TRADE_EVENTS_SCHEMA)
        conn.commit()
    return path


def write_account_snapshot(
    db_path: str | Path,
    snapshot: AccountSnapshot,
    *,
    snapshot_date: str | date | datetime | None = None,
) -> AccountSnapshotWriteResult:
    """Persist a canonical account snapshot plus daily NAV row."""

    path = ensure_account_ledger_schema(db_path)
    date_value = _date_text(snapshot_date or snapshot.as_of)
    as_of_text = snapshot.as_of.isoformat()
    account_key = snapshot.account_key
    nav_value = float(snapshot.net_liquidation or 0.0)
    cash_value = None if snapshot.cash is None else float(snapshot.cash)

    with closing(sqlite3.connect(path)) as conn:
        daily_pnl = _daily_pnl_from_previous_nav(
            conn,
            account_key=account_key,
            date_value=date_value,
            net_liquidation=nav_value,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO account_snapshots (
                snapshot_id,
                account_key,
                as_of,
                snapshot_date,
                account_id,
                broker,
                profile,
                environment,
                currency,
                net_liquidation,
                cash,
                buying_power,
                gross_position_value,
                margin_buffer,
                position_count,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                account_key,
                as_of_text,
                date_value,
                snapshot.account_id,
                snapshot.broker,
                snapshot.profile,
                snapshot.environment.value,
                snapshot.currency,
                nav_value,
                cash_value,
                _optional_float(snapshot.buying_power),
                _optional_float(snapshot.computed_gross_position_value),
                _optional_float(snapshot.margin_buffer),
                snapshot.position_count,
                _json(snapshot.metadata),
            ),
        )
        conn.execute(
            "DELETE FROM account_positions WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        )
        for position in snapshot.positions:
            conn.execute(
                """
                INSERT INTO account_positions (
                    snapshot_id,
                    account_key,
                    as_of,
                    snapshot_date,
                    account_id,
                    broker,
                    profile,
                    environment,
                    symbol,
                    asset_class,
                    quantity,
                    average_cost,
                    market_price,
                    market_value,
                    unrealized_pnl,
                    currency,
                    multiplier,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    account_key,
                    as_of_text,
                    date_value,
                    snapshot.account_id,
                    snapshot.broker,
                    snapshot.profile,
                    snapshot.environment.value,
                    position.symbol,
                    position.asset_class,
                    float(position.quantity),
                    _optional_float(position.average_cost),
                    _optional_float(position.market_price),
                    _optional_float(position.computed_market_value),
                    _optional_float(position.unrealized_pnl),
                    position.currency,
                    float(position.multiplier),
                    _json(position.metadata),
                ),
            )
        conn.execute(
            "DELETE FROM account_cash WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        )
        for cash in snapshot.cash_balances:
            conn.execute(
                """
                INSERT INTO account_cash (
                    snapshot_id,
                    account_key,
                    as_of,
                    snapshot_date,
                    account_id,
                    broker,
                    profile,
                    environment,
                    currency,
                    cash,
                    settled_cash,
                    buying_power,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    account_key,
                    as_of_text,
                    date_value,
                    snapshot.account_id,
                    snapshot.broker,
                    snapshot.profile,
                    snapshot.environment.value,
                    cash.currency,
                    float(cash.cash),
                    _optional_float(cash.settled_cash),
                    _optional_float(cash.buying_power),
                    _json(cash.metadata),
                ),
            )
        conn.execute(
            """
            INSERT INTO account_nav (
                date,
                account_key,
                account_id,
                broker,
                profile,
                environment,
                as_of,
                net_liquidation,
                cash,
                daily_pnl,
                position_count,
                snapshot_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, account_key) DO UPDATE SET
                account_id = excluded.account_id,
                broker = excluded.broker,
                profile = excluded.profile,
                environment = excluded.environment,
                as_of = excluded.as_of,
                net_liquidation = excluded.net_liquidation,
                cash = excluded.cash,
                daily_pnl = excluded.daily_pnl,
                position_count = excluded.position_count,
                snapshot_id = excluded.snapshot_id
            """,
            (
                date_value,
                account_key,
                snapshot.account_id,
                snapshot.broker,
                snapshot.profile,
                snapshot.environment.value,
                as_of_text,
                nav_value,
                cash_value,
                daily_pnl,
                snapshot.position_count,
                snapshot.snapshot_id,
            ),
        )
        conn.commit()

    return AccountSnapshotWriteResult(
        db_path=path,
        snapshot_id=snapshot.snapshot_id,
        account_key=account_key,
        snapshot_date=date_value,
        environment=snapshot.environment.value,
        profile=snapshot.profile,
        account_id=snapshot.account_id,
        position_rows=snapshot.position_count,
        cash_rows=len(snapshot.cash_balances),
        net_liquidation=nav_value,
        daily_pnl=daily_pnl,
    )


def load_latest_account_nav(
    db_path: str | Path,
    *,
    account_key: str | None = None,
    environment: str | None = None,
    profile: str | None = None,
) -> pd.DataFrame:
    columns = [
        "date",
        "account_key",
        "account_id",
        "broker",
        "profile",
        "environment",
        "as_of",
        "net_liquidation",
        "cash",
        "daily_pnl",
        "position_count",
        "snapshot_id",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)

    ensure_account_ledger_schema(path)
    where, params = _filters(
        account_key=account_key,
        environment=environment,
        profile=profile,
    )
    query = f"""
        SELECT {", ".join(columns)}
        FROM account_nav
        {where}
        ORDER BY as_of DESC
        LIMIT 1
    """
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(query, conn, params=params)


def load_account_nav_history(
    db_path: str | Path,
    *,
    account_key: str | None = None,
    environment: str | None = None,
    profile: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    columns = [
        "date",
        "account_key",
        "account_id",
        "broker",
        "profile",
        "environment",
        "as_of",
        "net_liquidation",
        "cash",
        "daily_pnl",
        "position_count",
        "snapshot_id",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)

    ensure_account_ledger_schema(path)
    where, params = _filters(
        account_key=account_key,
        environment=environment,
        profile=profile,
    )
    limit_clause = "" if limit is None else "LIMIT ?"
    query = f"""
        SELECT {", ".join(columns)}
        FROM account_nav
        {where}
        ORDER BY date DESC, as_of DESC
        {limit_clause}
    """
    query_params = params if limit is None else (*params, int(limit))
    with closing(sqlite3.connect(path)) as conn:
        frame = pd.read_sql(query, conn, params=query_params)

    if frame.empty:
        return frame
    return frame.sort_values(["date", "as_of"]).reset_index(drop=True)


def load_latest_account_positions(
    db_path: str | Path,
    *,
    account_key: str | None = None,
    environment: str | None = None,
    profile: str | None = None,
) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()

    ensure_account_ledger_schema(path)
    where, params = _filters(
        account_key=account_key,
        environment=environment,
        profile=profile,
        table_alias="s",
    )
    with closing(sqlite3.connect(path)) as conn:
        latest = conn.execute(
            f"""
            SELECT s.snapshot_id
            FROM account_snapshots s
            {where}
            ORDER BY s.as_of DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if latest is None:
            return pd.DataFrame()
        return pd.read_sql(
            """
            SELECT *
            FROM account_positions
            WHERE snapshot_id = ?
            ORDER BY ABS(COALESCE(market_value, quantity * COALESCE(market_price, 0) * multiplier)) DESC
            """,
            conn,
            params=(latest[0],),
        )


def _daily_pnl_from_previous_nav(
    conn: sqlite3.Connection,
    *,
    account_key: str,
    date_value: str,
    net_liquidation: float,
) -> float:
    previous = conn.execute(
        """
        SELECT net_liquidation
        FROM account_nav
        WHERE account_key = ? AND date < ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (account_key, date_value),
    ).fetchone()
    if previous is None or previous[0] is None:
        return 0.0
    return float(net_liquidation) - float(previous[0])


def _filters(
    *,
    account_key: str | None,
    environment: str | None,
    profile: str | None,
    table_alias: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    prefix = f"{table_alias}." if table_alias else ""
    clauses: list[str] = []
    params: list[Any] = []
    if account_key:
        clauses.append(f"{prefix}account_key = ?")
        params.append(account_key)
    if environment:
        clauses.append(f"{prefix}environment = ?")
        params.append(environment)
    if profile:
        clauses.append(f"{prefix}profile = ?")
        params.append(profile)
    return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(params))


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _redact_account(account_id: str | None) -> str:
    if not account_id:
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
