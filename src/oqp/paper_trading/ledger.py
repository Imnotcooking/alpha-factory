"""SQLite ledger for IBKR paper account monitoring."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import pandas as pd

from oqp.accounts import account_snapshot_from_ibkr_readonly, write_account_snapshot
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
    account_ledger_path: Path | None = None
    account_snapshot_id: str | None = None

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
            "account_ledger_path": None
            if self.account_ledger_path is None
            else str(self.account_ledger_path),
            "account_snapshot_id": self.account_snapshot_id,
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


@dataclass(frozen=True, slots=True)
class PaperOrderTicketWriteResult:
    db_path: Path
    order_ids: tuple[str, ...]

    @property
    def order_count(self) -> int:
        return len(self.order_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "order_count": self.order_count,
            "order_ids": list(self.order_ids),
        }


@dataclass(frozen=True, slots=True)
class PaperOrderTicketStatusUpdateResult:
    db_path: Path
    order_id: str
    previous_status: str
    new_status: str
    updated_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "order_id": self.order_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
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
    account_ledger_path: str | Path | None = None,
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
    account_snapshot_id = None
    account_ledger_written_path = None

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

    if account_ledger_path is not None:
        account_write = write_account_snapshot(
            account_ledger_path,
            account_snapshot_from_ibkr_readonly(
                snapshot,
                environment="paper",
                profile=profile,
                broker="ibkr",
                broker_label=broker_label,
                snapshot_date=date_value,
            ),
            snapshot_date=date_value,
        )
        account_snapshot_id = account_write.snapshot_id
        account_ledger_written_path = account_write.db_path

    return PaperSnapshotWriteResult(
        db_path=path,
        snapshot_id=snapshot_id,
        snapshot_date=date_value,
        account_id=account_id,
        position_rows=len(positions),
        net_liquidation=net_liquidation,
        cash=cash,
        daily_pnl=daily_pnl,
        account_ledger_path=account_ledger_written_path,
        account_snapshot_id=account_snapshot_id,
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


def write_paper_order_tickets(
    db_path: str | Path,
    tickets: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> PaperOrderTicketWriteResult:
    path = ensure_paper_trading_schema(db_path)
    if not tickets:
        return PaperOrderTicketWriteResult(db_path=path, order_ids=())

    with closing(sqlite3.connect(path)) as conn:
        for ticket in tickets:
            conn.execute(
                """
                INSERT OR REPLACE INTO paper_orders (
                    order_id,
                    created_at,
                    strategy_id,
                    symbol,
                    side,
                    quantity,
                    order_type,
                    limit_price,
                    status,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _required_text(ticket, "order_id"),
                    _required_text(ticket, "created_at"),
                    _optional_text(ticket.get("strategy_id")),
                    _required_text(ticket, "symbol"),
                    _required_text(ticket, "side"),
                    _float(ticket.get("quantity")),
                    _required_text(ticket, "order_type"),
                    _optional_float(ticket.get("limit_price")),
                    _required_text(ticket, "status"),
                    json.dumps(dict(ticket.get("metadata") or {}), sort_keys=True),
                ),
            )
        conn.commit()

    return PaperOrderTicketWriteResult(
        db_path=path,
        order_ids=tuple(str(ticket["order_id"]) for ticket in tickets),
    )


def load_latest_paper_orders(
    db_path: str | Path,
    *,
    limit: int = 50,
) -> pd.DataFrame:
    columns = [
        "order_id",
        "created_at",
        "strategy_id",
        "symbol",
        "side",
        "quantity",
        "order_type",
        "limit_price",
        "status",
        "metadata_json",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            """
            SELECT order_id, created_at, strategy_id, symbol, side, quantity,
                   order_type, limit_price, status, metadata_json
            FROM paper_orders
            ORDER BY created_at DESC, order_id DESC
            LIMIT ?
            """,
            conn,
            params=(max(int(limit), 1),),
        )


def load_paper_order_ticket(
    db_path: str | Path,
    order_id: str,
) -> dict[str, Any] | None:
    path = Path(db_path)
    if not path.exists():
        return None
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, created_at, strategy_id, symbol, side, quantity,
                   order_type, limit_price, status, metadata_json
            FROM paper_orders
            WHERE order_id = ?
            """,
            (str(order_id),),
        ).fetchone()
    if row is None:
        return None
    ticket = dict(row)
    ticket["metadata"] = _metadata_dict(ticket.get("metadata_json"))
    return ticket


def update_paper_order_ticket_status(
    db_path: str | Path,
    order_id: str,
    status: str,
    *,
    decided_at: str | datetime | None = None,
    decided_by: str | None = None,
    reason: str | None = None,
) -> PaperOrderTicketStatusUpdateResult:
    path = ensure_paper_trading_schema(db_path)
    new_status = str(status).strip()
    if not new_status:
        raise ValueError("status is required")
    timestamp = _datetime_text(decided_at or datetime.now(timezone.utc))

    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT order_id, status, metadata_json
            FROM paper_orders
            WHERE order_id = ?
            """,
            (str(order_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Paper order ticket not found: {order_id}")

        previous_status = str(row["status"])
        metadata = _metadata_dict(row["metadata_json"])
        metadata.update(
            {
                "previous_status": previous_status,
                "approval_status": new_status,
                "approval_decided_at": timestamp,
                "human_approval_required": True,
                "broker_submit_enabled": False,
            }
        )
        if decided_by:
            metadata["approval_decided_by"] = str(decided_by)
        if reason:
            metadata["approval_reason"] = str(reason)
        if new_status == "approved_for_submit":
            metadata["approved_at"] = timestamp
            if decided_by:
                metadata["approved_by"] = str(decided_by)
        elif new_status == "rejected":
            metadata["rejected_at"] = timestamp
            if decided_by:
                metadata["rejected_by"] = str(decided_by)

        conn.execute(
            """
            UPDATE paper_orders
            SET status = ?, metadata_json = ?
            WHERE order_id = ?
            """,
            (new_status, json.dumps(metadata, sort_keys=True), str(order_id)),
        )
        conn.commit()

    return PaperOrderTicketStatusUpdateResult(
        db_path=path,
        order_id=str(order_id),
        previous_status=previous_status,
        new_status=new_status,
        updated_at=timestamp,
        metadata=metadata,
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


def _datetime_text(value: str | datetime) -> str:
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    return str(value)


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _required_text(row: Mapping[str, Any], key: str) -> str:
    text = _optional_text(row.get(key))
    if text is None:
        raise ValueError(f"{key} is required")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _redact_account(account_id: str | None) -> str:
    if not account_id:
        return "n/a"
    text = str(account_id)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"
