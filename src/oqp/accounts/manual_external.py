"""Manual external holdings stored beside the unified account ledger.

These rows represent assets held away from the primary IBKR account. They are
kept separate from broker snapshots, then optionally layered into dashboard
views as external/manual positions.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_MANUAL_EXTERNAL_INPUT_PATH = (
    REPO_ROOT / "runtime" / "state" / "portfolio" / "manual_external_holdings.json"
)

MANUAL_EXTERNAL_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS manual_external_positions (
    position_id TEXT PRIMARY KEY,
    broker TEXT NOT NULL,
    account_id TEXT,
    environment TEXT NOT NULL,
    symbol TEXT NOT NULL,
    display_symbol TEXT,
    asset_class TEXT NOT NULL,
    quantity REAL NOT NULL,
    average_cost REAL,
    current_price REAL,
    local_cost_basis REAL,
    local_market_value REAL,
    local_unrealized_pnl REAL,
    currency TEXT NOT NULL,
    multiplier REAL NOT NULL DEFAULT 1.0,
    base_currency TEXT NOT NULL DEFAULT 'USD',
    fx_rate_to_base REAL,
    base_average_cost REAL,
    base_current_price REAL,
    base_cost_basis REAL,
    base_market_value REAL,
    base_unrealized_pnl REAL,
    underlying TEXT,
    expiry TEXT,
    option_type TEXT,
    strike REAL,
    side TEXT NOT NULL DEFAULT 'long',
    spread_group TEXT,
    quote_symbol TEXT,
    pricing_method TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    opened_at TEXT,
    updated_at TEXT NOT NULL,
    notes TEXT,
    metadata_json TEXT
)
"""

MANUAL_EXTERNAL_COLUMNS = [
    "position_id",
    "broker",
    "account_id",
    "environment",
    "symbol",
    "display_symbol",
    "asset_class",
    "quantity",
    "average_cost",
    "current_price",
    "local_cost_basis",
    "local_market_value",
    "local_unrealized_pnl",
    "currency",
    "multiplier",
    "base_currency",
    "fx_rate_to_base",
    "base_average_cost",
    "base_current_price",
    "base_cost_basis",
    "base_market_value",
    "base_unrealized_pnl",
    "underlying",
    "expiry",
    "option_type",
    "strike",
    "side",
    "spread_group",
    "quote_symbol",
    "pricing_method",
    "active",
    "opened_at",
    "updated_at",
    "notes",
    "metadata_json",
]

ACCOUNT_POSITION_COLUMNS = [
    "snapshot_id",
    "account_key",
    "as_of",
    "snapshot_date",
    "account_id",
    "broker",
    "profile",
    "environment",
    "symbol",
    "asset_class",
    "quantity",
    "average_cost",
    "market_price",
    "market_value",
    "unrealized_pnl",
    "currency",
    "multiplier",
    "metadata_json",
]


def ensure_manual_external_schema(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(MANUAL_EXTERNAL_POSITIONS_SCHEMA)
        conn.commit()
    return path


def load_manual_external_positions_file(path: str | Path = DEFAULT_MANUAL_EXTERNAL_INPUT_PATH) -> list[dict[str, Any]]:
    """Load manual external holdings from the ignored runtime JSON file."""

    input_path = Path(path)
    if not input_path.exists():
        return []
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = payload.get("positions", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Manual external holdings JSON must be a list or an object with a positions list.")
    return [row for row in rows if isinstance(row, dict)]


def sync_manual_external_positions_from_json(
    db_path: str | Path,
    *,
    json_path: str | Path = DEFAULT_MANUAL_EXTERNAL_INPUT_PATH,
    replace: bool = True,
    preserve_refreshed_pricing: bool = True,
) -> int:
    """Sync the ignored JSON source into SQLite for dashboard consumption.

    With ``replace=True``, active rows missing from the JSON are marked inactive.
    This makes the JSON behave like the current source-of-truth file.
    """

    rows = load_manual_external_positions_file(json_path)
    if not rows:
        ensure_manual_external_schema(db_path)
        return 0

    normalized = [normalize_manual_external_position(row) for row in rows]
    if preserve_refreshed_pricing:
        normalized = _preserve_refreshed_pricing(db_path, normalized)
    written = upsert_manual_external_positions(db_path, normalized)
    if replace:
        _deactivate_missing_positions(db_path, normalized)
    return written


def _preserve_refreshed_pricing(
    db_path: str | Path,
    normalized_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep refreshed market prices when JSON only contains cost placeholders."""

    path = Path(db_path)
    if not path.exists():
        return normalized_rows
    existing = load_manual_external_positions(path, active_only=False)
    if existing.empty:
        return normalized_rows

    by_id = {
        str(row.get("position_id")): row
        for row in existing.to_dict("records")
        if row.get("position_id")
    }
    merged: list[dict[str, Any]] = []
    for row in normalized_rows:
        current = dict(row)
        existing_row = by_id.get(str(current.get("position_id")))
        if existing_row is None:
            merged.append(current)
            continue

        if _optional_float(current.get("fx_rate_to_base")) is None:
            existing_fx = _optional_float(existing_row.get("fx_rate_to_base"))
            if existing_fx is not None:
                current["fx_rate_to_base"] = existing_fx
                if existing_row.get("base_currency"):
                    current["base_currency"] = existing_row.get("base_currency")

        incoming_method = str(current.get("pricing_method") or "").lower()
        existing_method = str(existing_row.get("pricing_method") or "").lower()
        incoming_is_placeholder = incoming_method in {"", "manual_cost", "manual_cash", "manual_cost_fallback"}
        existing_is_refreshed = existing_method not in {"", "manual_cost", "manual_cash", "manual_cost_fallback"}
        if incoming_is_placeholder and existing_is_refreshed:
            for field in ("current_price", "fx_rate_to_base", "base_currency", "metadata_json"):
                value = existing_row.get(field)
                if value is not None and not pd.isna(value):
                    current[field] = value
            for derived_field in (
                "local_market_value",
                "local_unrealized_pnl",
                "base_current_price",
                "base_market_value",
                "base_unrealized_pnl",
            ):
                current[derived_field] = None
            current["pricing_method"] = existing_row.get("pricing_method")
        merged.append(current)
    return merged


def normalize_manual_external_position(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one manual holding input into the SQL table shape."""

    now = utc_timestamp()
    position_id = str(row.get("position_id") or row.get("id") or "").strip()
    if not position_id:
        raise ValueError("manual external position requires position_id")

    currency = str(row.get("currency") or "USD").upper().strip()
    base_currency = str(row.get("base_currency") or "USD").upper().strip()
    quantity = _float(row.get("quantity"), 0.0)
    multiplier = _float(row.get("multiplier"), 1.0) or 1.0
    average_cost = _optional_float(row.get("average_cost"))
    current_price = _optional_float(row.get("current_price"))
    if current_price is None:
        current_price = average_cost

    local_cost_basis = _optional_float(row.get("local_cost_basis"))
    if local_cost_basis is None and average_cost is not None:
        local_cost_basis = quantity * average_cost * multiplier
    local_market_value = _optional_float(row.get("local_market_value"))
    if local_market_value is None and current_price is not None:
        local_market_value = quantity * current_price * multiplier
    local_unrealized_pnl = _optional_float(row.get("local_unrealized_pnl"))
    if local_unrealized_pnl is None and local_market_value is not None and local_cost_basis is not None:
        local_unrealized_pnl = local_market_value - local_cost_basis

    fx_rate = _optional_float(row.get("fx_rate_to_base"))
    if fx_rate is None and currency == base_currency:
        fx_rate = 1.0

    base_average_cost = _optional_float(row.get("base_average_cost"))
    base_current_price = _optional_float(row.get("base_current_price"))
    base_cost_basis = _optional_float(row.get("base_cost_basis"))
    base_market_value = _optional_float(row.get("base_market_value"))
    base_unrealized_pnl = _optional_float(row.get("base_unrealized_pnl"))
    if fx_rate is not None:
        base_average_cost = base_average_cost if base_average_cost is not None else _multiply(average_cost, fx_rate)
        base_current_price = base_current_price if base_current_price is not None else _multiply(current_price, fx_rate)
        base_cost_basis = base_cost_basis if base_cost_basis is not None else _multiply(local_cost_basis, fx_rate)
        base_market_value = base_market_value if base_market_value is not None else _multiply(local_market_value, fx_rate)
        base_unrealized_pnl = base_unrealized_pnl if base_unrealized_pnl is not None else _multiply(local_unrealized_pnl, fx_rate)

    metadata = _json_dict(row.get("metadata_json"))
    metadata.update(dict(row.get("metadata") or {}))
    metadata.update(
        {
            "source": "manual_external_position",
            "native_currency": currency,
            "local_average_cost": average_cost,
            "local_current_price": current_price,
            "local_cost_basis": local_cost_basis,
            "local_market_value": local_market_value,
            "local_unrealized_pnl": local_unrealized_pnl,
            "base_currency": base_currency,
            "fx_rate_to_base": fx_rate,
            "pricing_method": row.get("pricing_method") or "manual_cost",
        }
    )

    return {
        "position_id": position_id,
        "broker": str(row.get("broker") or "external_manual").strip(),
        "account_id": row.get("account_id") or "external",
        "environment": str(row.get("environment") or "live").strip().lower(),
        "symbol": str(row.get("symbol") or position_id).strip(),
        "display_symbol": row.get("display_symbol") or row.get("symbol") or position_id,
        "asset_class": str(row.get("asset_class") or "equity").strip().lower(),
        "quantity": quantity,
        "average_cost": average_cost,
        "current_price": current_price,
        "local_cost_basis": local_cost_basis,
        "local_market_value": local_market_value,
        "local_unrealized_pnl": local_unrealized_pnl,
        "currency": currency,
        "multiplier": multiplier,
        "base_currency": base_currency,
        "fx_rate_to_base": fx_rate,
        "base_average_cost": base_average_cost,
        "base_current_price": base_current_price,
        "base_cost_basis": base_cost_basis,
        "base_market_value": base_market_value,
        "base_unrealized_pnl": base_unrealized_pnl,
        "underlying": row.get("underlying"),
        "expiry": row.get("expiry"),
        "option_type": row.get("option_type"),
        "strike": _optional_float(row.get("strike")),
        "side": str(row.get("side") or "long").strip().lower(),
        "spread_group": row.get("spread_group"),
        "quote_symbol": row.get("quote_symbol"),
        "pricing_method": row.get("pricing_method") or "manual_cost",
        "active": 1 if bool(row.get("active", True)) else 0,
        "opened_at": row.get("opened_at"),
        "updated_at": row.get("updated_at") or now,
        "notes": row.get("notes"),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def upsert_manual_external_positions(
    db_path: str | Path,
    rows: list[dict[str, Any]],
) -> int:
    path = ensure_manual_external_schema(db_path)
    normalized = [normalize_manual_external_position(row) for row in rows]
    if not normalized:
        return 0
    placeholders = ", ".join("?" for _ in MANUAL_EXTERNAL_COLUMNS)
    updates = ", ".join(
        f"{column}=excluded.{column}"
        for column in MANUAL_EXTERNAL_COLUMNS
        if column != "position_id"
    )
    with closing(sqlite3.connect(path)) as conn:
        conn.executemany(
            f"""
            INSERT INTO manual_external_positions ({", ".join(MANUAL_EXTERNAL_COLUMNS)})
            VALUES ({placeholders})
            ON CONFLICT(position_id) DO UPDATE SET {updates}
            """,
            [tuple(row.get(column) for column in MANUAL_EXTERNAL_COLUMNS) for row in normalized],
        )
        conn.commit()
    return len(normalized)


def _deactivate_missing_positions(
    db_path: str | Path,
    normalized_rows: list[dict[str, Any]],
) -> None:
    path = ensure_manual_external_schema(db_path)
    by_environment: dict[str, list[str]] = {}
    for row in normalized_rows:
        by_environment.setdefault(str(row.get("environment") or "live"), []).append(str(row["position_id"]))
    with closing(sqlite3.connect(path)) as conn:
        for environment, position_ids in by_environment.items():
            placeholders = ", ".join("?" for _ in position_ids)
            conn.execute(
                f"""
                UPDATE manual_external_positions
                SET active = 0, updated_at = ?
                WHERE environment = ?
                  AND active = 1
                  AND position_id NOT IN ({placeholders})
                """,
                (utc_timestamp(), environment, *position_ids),
            )
        conn.commit()


def load_manual_external_positions(
    db_path: str | Path,
    *,
    environment: str = "live",
    active_only: bool = True,
) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=MANUAL_EXTERNAL_COLUMNS)
    ensure_manual_external_schema(path)
    where = ["environment = ?"]
    params: list[Any] = [environment]
    if active_only:
        where.append("active = 1")
    with closing(sqlite3.connect(path)) as conn:
        frame = pd.read_sql(
            f"""
            SELECT {", ".join(MANUAL_EXTERNAL_COLUMNS)}
            FROM manual_external_positions
            WHERE {" AND ".join(where)}
            ORDER BY active DESC, broker, symbol
            """,
            conn,
            params=params,
        )
    return frame if not frame.empty else pd.DataFrame(columns=MANUAL_EXTERNAL_COLUMNS)


def load_manual_external_positions_as_account_positions(
    db_path: str | Path,
    *,
    environment: str = "live",
    profile: str = "manual_external",
) -> pd.DataFrame:
    """Return active manual rows in the account_positions table shape."""

    manual = load_manual_external_positions(db_path, environment=environment)
    if manual.empty:
        return pd.DataFrame(columns=ACCOUNT_POSITION_COLUMNS)

    now = utc_timestamp()
    today = now[:10]
    rows: list[dict[str, Any]] = []
    for row in manual.to_dict("records"):
        metadata = _json_dict(row.get("metadata_json"))
        has_base_value = row.get("base_market_value") is not None and not pd.isna(row.get("base_market_value"))
        value_currency = str(row.get("base_currency") or "USD") if has_base_value else str(row.get("currency") or "USD")
        average_cost = row.get("base_average_cost") if has_base_value else row.get("average_cost")
        market_price = row.get("base_current_price") if has_base_value else row.get("current_price")
        market_value = row.get("base_market_value") if has_base_value else row.get("local_market_value")
        unrealized_pnl = row.get("base_unrealized_pnl") if has_base_value else row.get("local_unrealized_pnl")
        metadata.update(
            {
                "position_id": row.get("position_id"),
                "display_symbol": row.get("display_symbol"),
                "manual_external": True,
                "value_currency": value_currency,
                "valuation_base_ready": bool(has_base_value),
                "underlying": row.get("underlying"),
                "expiry": row.get("expiry"),
                "option_type": row.get("option_type"),
                "strike": row.get("strike"),
                "side": row.get("side"),
                "spread_group": row.get("spread_group"),
                "quote_symbol": row.get("quote_symbol"),
                "notes": row.get("notes"),
            }
        )
        broker = str(row.get("broker") or "external_manual")
        account_id = str(row.get("account_id") or "external")
        rows.append(
            {
                "snapshot_id": f"manual-external-{today}",
                "account_key": f"{environment}:{broker}:{profile}:{account_id}",
                "as_of": row.get("updated_at") or now,
                "snapshot_date": today,
                "account_id": account_id,
                "broker": broker,
                "profile": profile,
                "environment": environment,
                "symbol": row.get("symbol"),
                "asset_class": row.get("asset_class"),
                "quantity": row.get("quantity"),
                "average_cost": average_cost,
                "market_price": market_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "currency": value_currency,
                "multiplier": row.get("multiplier") or 1.0,
                "metadata_json": json.dumps(metadata, sort_keys=True),
            }
        )
    return pd.DataFrame(rows, columns=ACCOUNT_POSITION_COLUMNS)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _multiply(value: float | None, factor: float) -> float | None:
    return None if value is None else float(value) * float(factor)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
