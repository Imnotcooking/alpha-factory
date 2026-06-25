"""SQLite ledger for IBKR paper account monitoring."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from oqp.brokers import IBKRReadOnlyPortfolioSnapshot


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER_TRADING_DB_PATH = (
    REPO_ROOT / "data" / "paper_trading" / "paper_trading.db"
)


PAPER_ACCOUNT_SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_account_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    as_of TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    profile TEXT NOT NULL,
    currency TEXT NOT NULL,
    net_liquidation REAL,
    cash REAL,
    buying_power REAL,
    gross_position_value REAL,
    margin_buffer REAL,
    position_count INTEGER NOT NULL
)
"""

PAPER_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_positions (
    snapshot_id TEXT NOT NULL,
    as_of TEXT NOT NULL,
    account_id TEXT,
    broker TEXT NOT NULL,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_cost REAL,
    market_price REAL,
    market_value REAL,
    unrealized_pnl REAL,
    currency TEXT NOT NULL,
    multiplier REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (snapshot_id, broker, symbol, asset_type)
)
"""

PAPER_NAV_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_nav (
    date TEXT PRIMARY KEY,
    account_id TEXT,
    as_of TEXT NOT NULL,
    net_liquidation REAL NOT NULL,
    cash REAL,
    daily_pnl REAL,
    position_count INTEGER NOT NULL,
    snapshot_id TEXT NOT NULL
)
"""

PAPER_ORDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_orders (
    order_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    strategy_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    order_type TEXT NOT NULL,
    limit_price REAL,
    status TEXT NOT NULL,
    metadata_json TEXT
)
"""

PAPER_FILLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT,
    executed_at TEXT NOT NULL,
    strategy_id TEXT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_price REAL NOT NULL,
    commission REAL,
    currency TEXT,
    metadata_json TEXT
)
"""

PAPER_EXECUTION_REVIEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_execution_reviews (
    review_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    reviewed_at TEXT NOT NULL,
    decision TEXT NOT NULL,
    estimated_notional REAL,
    order_count INTEGER NOT NULL,
    checks_json TEXT NOT NULL,
    message TEXT
)
"""


@dataclass(frozen=True, slots=True)
class PaperSnapshotWriteResult:
    db_path: Path
    snapshot_id: str
    snapshot_date: str
    account_id: str | None
    position_rows: int
    net_liquidation: float
    cash: float
    daily_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "snapshot_id": self.snapshot_id,
            "snapshot_date": self.snapshot_date,
            "account_id": _redact_account(self.account_id),
            "position_rows": self.position_rows,
            "net_liquidation": self.net_liquidation,
            "cash": self.cash,
            "daily_pnl": self.daily_pnl,
        }


@dataclass(frozen=True, slots=True)
class PaperExecutionReviewWriteResult:
    db_path: Path
    review_id: str
    proposal_id: str
    decision: str
    order_count: int
    estimated_notional: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "review_id": self.review_id,
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "order_count": self.order_count,
            "estimated_notional": self.estimated_notional,
        }


def default_paper_trading_ledger_path() -> Path:
    return DEFAULT_PAPER_TRADING_DB_PATH


def ensure_paper_trading_schema(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(PAPER_ACCOUNT_SNAPSHOTS_SCHEMA)
        conn.execute(PAPER_POSITIONS_SCHEMA)
        conn.execute(PAPER_NAV_SCHEMA)
        conn.execute(PAPER_ORDERS_SCHEMA)
        conn.execute(PAPER_FILLS_SCHEMA)
        conn.execute(PAPER_EXECUTION_REVIEWS_SCHEMA)
        conn.commit()
    return path


def write_paper_snapshot(
    db_path: str | Path,
    snapshot: IBKRReadOnlyPortfolioSnapshot,
    *,
    snapshot_date: str | date | datetime | None = None,
    broker_label: str = "IBKR Paper",
    profile: str = "ibkr_paper_readonly",
) -> PaperSnapshotWriteResult:
    """Persist one read-only IBKR paper snapshot and daily NAV observation."""

    if snapshot.error:
        raise ValueError(f"Paper snapshot contains an error: {snapshot.error}")

    path = ensure_paper_trading_schema(db_path)
    as_of = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    date_value = _date_text(snapshot_date or date.today())
    account_id = snapshot.health.account_id
    snapshot_id = _snapshot_id(account_id=account_id, as_of=as_of)
    metrics = snapshot.metrics
    net_liquidation = _float(metrics.get("Total_NAV_USD"))
    cash = _float(metrics.get("Available_Cash_USD"))
    margin_buffer = _float(metrics.get("Margin_Buffer_USD"))
    positions = [_position_row(row) for row in snapshot.position_rows]

    with closing(sqlite3.connect(path)) as conn:
        daily_pnl = _daily_pnl_from_previous_nav(conn, date_value, net_liquidation)
        conn.execute(
            """
            INSERT INTO paper_account_snapshots (
                snapshot_id,
                as_of,
                account_id,
                broker,
                profile,
                currency,
                net_liquidation,
                cash,
                buying_power,
                gross_position_value,
                margin_buffer,
                position_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                as_of,
                account_id,
                broker_label,
                profile,
                "USD",
                net_liquidation,
                cash,
                margin_buffer,
                None,
                margin_buffer,
                len(positions),
            ),
        )
        conn.execute(
            """
            INSERT INTO paper_nav (
                date,
                account_id,
                as_of,
                net_liquidation,
                cash,
                daily_pnl,
                position_count,
                snapshot_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                account_id = excluded.account_id,
                as_of = excluded.as_of,
                net_liquidation = excluded.net_liquidation,
                cash = excluded.cash,
                daily_pnl = excluded.daily_pnl,
                position_count = excluded.position_count,
                snapshot_id = excluded.snapshot_id
            """,
            (
                date_value,
                account_id,
                as_of,
                net_liquidation,
                cash,
                daily_pnl,
                len(positions),
                snapshot_id,
            ),
        )
        for position in positions:
            conn.execute(
                """
                INSERT INTO paper_positions (
                    snapshot_id,
                    as_of,
                    account_id,
                    broker,
                    symbol,
                    asset_type,
                    quantity,
                    average_cost,
                    market_price,
                    market_value,
                    unrealized_pnl,
                    currency,
                    multiplier
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    as_of,
                    account_id,
                    broker_label,
                    position["symbol"],
                    position["asset_type"],
                    position["quantity"],
                    position["average_cost"],
                    position["market_price"],
                    position["market_value"],
                    position["unrealized_pnl"],
                    position["currency"],
                    position["multiplier"],
                ),
            )
        conn.commit()

    return PaperSnapshotWriteResult(
        db_path=path,
        snapshot_id=snapshot_id,
        snapshot_date=date_value,
        account_id=account_id,
        position_rows=len(positions),
        net_liquidation=net_liquidation,
        cash=cash,
        daily_pnl=daily_pnl,
    )


def load_latest_paper_nav(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "account_id",
                "as_of",
                "net_liquidation",
                "cash",
                "daily_pnl",
                "position_count",
                "snapshot_id",
            ]
        )
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            """
            SELECT date, account_id, as_of, net_liquidation, cash, daily_pnl,
                   position_count, snapshot_id
            FROM paper_nav
            ORDER BY date DESC
            LIMIT 1
            """,
            conn,
        )


def load_latest_paper_positions(db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        latest = conn.execute(
            """
            SELECT snapshot_id
            FROM paper_account_snapshots
            ORDER BY as_of DESC
            LIMIT 1
            """
        ).fetchone()
        if latest is None:
            return pd.DataFrame()
        return pd.read_sql(
            """
            SELECT *
            FROM paper_positions
            WHERE snapshot_id = ?
            ORDER BY ABS(quantity * COALESCE(market_price, 0) * multiplier) DESC
            """,
            conn,
            params=(latest[0],),
        )


def load_latest_paper_execution_reviews(
    db_path: str | Path,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    columns = [
        "review_id",
        "proposal_id",
        "reviewed_at",
        "decision",
        "estimated_notional",
        "order_count",
        "checks_json",
        "message",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            """
            SELECT review_id, proposal_id, reviewed_at, decision, estimated_notional,
                   order_count, checks_json, message
            FROM paper_execution_reviews
            ORDER BY reviewed_at DESC
            LIMIT ?
            """,
            conn,
            params=(max(int(limit), 1),),
        )


def write_paper_execution_review(
    db_path: str | Path,
    *,
    proposal_id: str,
    decision: str,
    checks: list[dict[str, Any]],
    estimated_notional: float | None,
    order_count: int,
    message: str | None = None,
    reviewed_at: datetime | None = None,
) -> PaperExecutionReviewWriteResult:
    path = ensure_paper_trading_schema(db_path)
    timestamp = (reviewed_at or datetime.now(timezone.utc)).replace(microsecond=0)
    reviewed_at_text = timestamp.isoformat()
    review_id = f"paper-review-{proposal_id}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"

    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_execution_reviews (
                review_id,
                proposal_id,
                reviewed_at,
                decision,
                estimated_notional,
                order_count,
                checks_json,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                proposal_id,
                reviewed_at_text,
                decision,
                None if estimated_notional is None else float(estimated_notional),
                int(order_count),
                json.dumps(checks, sort_keys=True),
                message,
            ),
        )
        conn.commit()

    return PaperExecutionReviewWriteResult(
        db_path=path,
        review_id=review_id,
        proposal_id=proposal_id,
        decision=decision,
        order_count=int(order_count),
        estimated_notional=estimated_notional,
    )


def paper_order_notional_today(db_path: str | Path, *, today: str | None = None) -> float:
    path = Path(db_path)
    if not path.exists():
        return 0.0
    ensure_paper_trading_schema(path)
    date_prefix = today or date.today().isoformat()
    with closing(sqlite3.connect(path)) as conn:
        value = conn.execute(
            """
            SELECT COALESCE(SUM(quantity * COALESCE(limit_price, 0)), 0)
            FROM paper_orders
            WHERE created_at >= ?
            """,
            (date_prefix,),
        ).fetchone()[0]
    return _float(value)


def _position_row(row: dict[str, Any]) -> dict[str, Any]:
    quantity = _float(row.get("Shares"))
    market_price = _float(row.get("Broker_Price"))
    multiplier = _float(row.get("Multiplier"), default=1.0) or 1.0
    return {
        "symbol": str(row.get("Ticker", "")).strip(),
        "asset_type": str(row.get("AssetType", "Equity")).strip() or "Equity",
        "quantity": quantity,
        "average_cost": _float(row.get("AvgPrice")),
        "market_price": market_price,
        "market_value": quantity * market_price * multiplier,
        "unrealized_pnl": _float(row.get("Broker_PnL")),
        "currency": str(row.get("Currency", "USD")).strip() or "USD",
        "multiplier": multiplier,
    }


def _daily_pnl_from_previous_nav(
    conn: sqlite3.Connection,
    date_value: str,
    net_liquidation: float,
) -> float:
    previous = conn.execute(
        """
        SELECT net_liquidation
        FROM paper_nav
        WHERE date < ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (date_value,),
    ).fetchone()
    if previous is None or previous[0] is None:
        return 0.0
    return float(net_liquidation) - float(previous[0])


def _snapshot_id(*, account_id: str | None, as_of: str) -> str:
    account = (account_id or "paper").replace(" ", "_")
    compact = (
        as_of.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace("+0000", "Z")
    )
    return f"paper-{account}-{compact}-{uuid4().hex[:8]}"


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _redact_account(account_id: str | None) -> str:
    if not account_id:
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
