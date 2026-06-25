"""SQLite-backed portfolio ledger helpers."""

from __future__ import annotations

import sqlite3
import os
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT, legacy_middle_office_root
from oqp.portfolio.snapshots import LIVE_POSITION_COLUMNS


DEFAULT_PORTFOLIO_DB_PATH = (
    REPO_ROOT / "runtime" / "db" / "portfolio" / "portfolio_ledger.db"
)
LEGACY_MIDDLE_OFFICE_DB_PATH = (
    legacy_middle_office_root() / "Portfolio" / "clean_data" / "macro_terminal.db"
)
DEFAULT_MIDDLE_OFFICE_DB_PATH = DEFAULT_PORTFOLIO_DB_PATH

HISTORICAL_NAV_SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_nav (
    date TEXT PRIMARY KEY,
    total_net_worth REAL,
    total_cash REAL,
    portfolio_beta REAL,
    daily_pnl REAL
)
"""

LIVE_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_positions (
    date TEXT,
    broker TEXT,
    ticker TEXT,
    asset_type TEXT,
    shares REAL,
    avg_cost REAL,
    current_price REAL,
    unrealized_pnl REAL,
    currency TEXT,
    delta REAL DEFAULT 1.0,
    gamma REAL DEFAULT 0.0,
    PRIMARY KEY (date, broker, ticker)
)
"""

HISTORICAL_NAV_COLUMNS = [
    "date",
    "total_net_worth",
    "total_cash",
    "portfolio_beta",
    "daily_pnl",
]

LEGACY_TO_LIVE_POSITION_COLUMNS = {
    "Broker": "broker",
    "Ticker": "ticker",
    "AssetType": "asset_type",
    "Shares": "shares",
    "AvgPrice": "avg_cost",
    "Broker_Price": "current_price",
    "Broker_PnL": "unrealized_pnl",
    "Currency": "currency",
}


def default_portfolio_ledger_path() -> Path:
    configured = os.getenv("OQP_PORTFOLIO_LEDGER_PATH")
    return Path(configured).expanduser() if configured else DEFAULT_PORTFOLIO_DB_PATH


def default_middle_office_ledger_path() -> Path:
    """Backward-compatible alias for the unified portfolio ledger path."""

    return default_portfolio_ledger_path()


def legacy_middle_office_ledger_path() -> Path:
    return LEGACY_MIDDLE_OFFICE_DB_PATH



def ensure_portfolio_ledger_schema(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(HISTORICAL_NAV_SCHEMA)
        conn.execute(LIVE_POSITIONS_SCHEMA)
        conn.commit()
    return path


def normalize_live_positions_frame(
    positions: pd.DataFrame,
    *,
    snapshot_date: str | date | datetime | None = None,
) -> pd.DataFrame:
    """Return a frame that matches the shared `live_positions` table schema."""

    df = positions.copy()
    df = df.rename(columns=LEGACY_TO_LIVE_POSITION_COLUMNS)

    if snapshot_date is not None:
        df["date"] = _date_text(snapshot_date)
    elif "date" not in df.columns:
        df["date"] = _date_text(date.today())

    defaults: dict[str, Any] = {
        "broker": "",
        "ticker": "",
        "asset_type": "Equity",
        "shares": 0.0,
        "avg_cost": 0.0,
        "current_price": 0.0,
        "unrealized_pnl": 0.0,
        "currency": "USD",
        "delta": 1.0,
        "gamma": 0.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    numeric_columns = [
        "shares",
        "avg_cost",
        "current_price",
        "unrealized_pnl",
        "delta",
        "gamma",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(defaults[column])

    for column in ("date", "broker", "ticker", "asset_type", "currency"):
        df[column] = df[column].fillna(defaults.get(column, "")).astype(str).str.strip()

    df = df[df["broker"].ne("") & df["ticker"].ne("")]
    return df[LIVE_POSITION_COLUMNS]


def write_live_positions_frame(
    db_path: str | Path,
    positions: pd.DataFrame,
    *,
    snapshot_date: str | date | datetime,
    replace_date: bool = True,
) -> int:
    """Write one date's portfolio positions to the shared live-position ledger."""

    path = ensure_portfolio_ledger_schema(db_path)
    live_positions = normalize_live_positions_frame(
        positions,
        snapshot_date=snapshot_date,
    )
    if live_positions.empty:
        return 0

    date_value = _date_text(snapshot_date)
    with closing(sqlite3.connect(path)) as conn:
        if replace_date:
            conn.execute("DELETE FROM live_positions WHERE date = ?", (date_value,))
        live_positions.to_sql("live_positions", conn, if_exists="append", index=False)
        conn.commit()
    return int(len(live_positions))


def load_latest_live_positions(db_path: str | Path) -> pd.DataFrame:
    """Load the newest available live-position snapshot."""

    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=LIVE_POSITION_COLUMNS)

    ensure_portfolio_ledger_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        latest = pd.read_sql(
            "SELECT MAX(date) AS last_date FROM live_positions",
            conn,
        )
        if latest.empty or pd.isna(latest.iloc[0]["last_date"]):
            return pd.DataFrame(columns=LIVE_POSITION_COLUMNS)

        query_date = str(latest.iloc[0]["last_date"])
        return pd.read_sql(
            "SELECT * FROM live_positions WHERE date = ? ORDER BY broker, ticker",
            conn,
            params=(query_date,),
        )


def write_historical_nav(
    db_path: str | Path,
    *,
    snapshot_date: str | date | datetime,
    total_net_worth: float,
    total_cash: float | None = None,
    portfolio_beta: float | None = None,
    daily_pnl: float | None = None,
) -> None:
    """Upsert one daily NAV observation into the portfolio ledger."""

    path = ensure_portfolio_ledger_schema(db_path)
    date_value = _date_text(snapshot_date)
    nav_value = float(total_net_worth)

    with closing(sqlite3.connect(path)) as conn:
        pnl_value = (
            float(daily_pnl)
            if daily_pnl is not None
            else _daily_pnl_from_previous_nav(conn, date_value, nav_value)
        )
        conn.execute(
            """
            INSERT INTO historical_nav (
                date,
                total_net_worth,
                total_cash,
                portfolio_beta,
                daily_pnl
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_net_worth = excluded.total_net_worth,
                total_cash = excluded.total_cash,
                portfolio_beta = excluded.portfolio_beta,
                daily_pnl = excluded.daily_pnl
            """,
            (
                date_value,
                nav_value,
                _optional_float(total_cash),
                _optional_float(portfolio_beta),
                pnl_value,
            ),
        )
        conn.commit()


def load_historical_nav(db_path: str | Path) -> pd.DataFrame:
    """Load all stored NAV observations ordered by date."""

    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORICAL_NAV_COLUMNS)

    ensure_portfolio_ledger_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            """
            SELECT date, total_net_worth, total_cash, portfolio_beta, daily_pnl
            FROM historical_nav
            ORDER BY date
            """,
            conn,
        )


def compute_nav_drawdowns(nav: pd.DataFrame) -> pd.DataFrame:
    """Add equity-peak and drawdown columns to a historical NAV frame."""

    if nav.empty:
        return pd.DataFrame(
            columns=HISTORICAL_NAV_COLUMNS
            + ["equity_peak", "drawdown", "drawdown_pct"]
        )

    out = nav.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["total_net_worth"] = pd.to_numeric(
        out["total_net_worth"], errors="coerce"
    ).fillna(0.0)
    out = out.dropna(subset=["date"]).sort_values("date")
    out["equity_peak"] = out["total_net_worth"].cummax()
    out["drawdown"] = out["total_net_worth"] - out["equity_peak"]
    out["drawdown_pct"] = (
        out["drawdown"] / out["equity_peak"].replace({0.0: pd.NA})
    ).fillna(0.0)
    return out


def _daily_pnl_from_previous_nav(
    conn: sqlite3.Connection,
    date_value: str,
    total_net_worth: float,
) -> float:
    previous = conn.execute(
        """
        SELECT total_net_worth
        FROM historical_nav
        WHERE date < ?
        ORDER BY date DESC
        LIMIT 1
        """,
        (date_value,),
    ).fetchone()
    if previous is None or previous[0] is None:
        return 0.0
    return float(total_net_worth) - float(previous[0])


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _date_text(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
