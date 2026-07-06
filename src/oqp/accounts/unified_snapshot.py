"""Build unified account snapshots from broker and external/manual holdings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.accounts.ledger import (
    AccountSnapshotWriteResult,
    load_latest_account_nav,
    load_latest_account_positions,
    write_account_snapshot,
)
from oqp.accounts.manual_external import load_manual_external_positions_as_account_positions
from oqp.accounts.models import (
    AccountEnvironment,
    AccountSnapshot,
    CashSnapshot,
    PositionSnapshot,
    account_timestamp,
)


UNIFIED_LIVE_PROFILE = "unified_live"
UNIFIED_LIVE_BROKER = "unified"
UNIFIED_LIVE_ACCOUNT_ID = "unified_live"
DEFAULT_LIVE_BROKER_PROFILE = "ibkr_live_readonly"


@dataclass(frozen=True, slots=True)
class UnifiedSnapshotResult:
    write_result: AccountSnapshotWriteResult
    broker_rows: int
    manual_rows: int
    currency: str
    manual_usd_value: float
    manual_usd_cash: float
    excluded_manual_rows: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.write_result.to_dict()
        payload.update(
            {
                "broker_rows": self.broker_rows,
                "manual_rows": self.manual_rows,
                "currency": self.currency,
                "manual_usd_value": self.manual_usd_value,
                "manual_usd_cash": self.manual_usd_cash,
                "excluded_manual_rows": self.excluded_manual_rows,
            }
        )
        return payload


def materialize_unified_live_account_snapshot(
    db_path: str | Path,
    *,
    broker_profile: str = DEFAULT_LIVE_BROKER_PROFILE,
    unified_profile: str = UNIFIED_LIVE_PROFILE,
) -> UnifiedSnapshotResult:
    """Write a combined live snapshot for dashboard/history consumption.

    The raw broker snapshot remains unchanged. This writes a separate
    ``unified_live`` profile containing the latest IBKR rows plus active
    external/manual rows that have USD-ready values.
    """

    latest_broker_nav = load_latest_account_nav(
        db_path,
        environment=AccountEnvironment.LIVE.value,
        profile=broker_profile,
    )
    broker_positions = load_latest_account_positions(
        db_path,
        environment=AccountEnvironment.LIVE.value,
        profile=broker_profile,
    )
    manual_positions = load_manual_external_positions_as_account_positions(
        db_path,
        environment=AccountEnvironment.LIVE.value,
    )

    target_currency = _latest_currency(latest_broker_nav) or "USD"
    manual_target = _manual_rows_for_currency(manual_positions, target_currency)
    excluded_manual_rows = max(len(manual_positions) - len(manual_target), 0)
    broker_nav_value = _latest_float(latest_broker_nav, "net_liquidation") or 0.0
    broker_cash_value = _latest_float(latest_broker_nav, "cash") or 0.0
    manual_target_value = _sum_numeric(manual_target, "market_value")
    manual_target_cash = _sum_numeric(
        manual_target.loc[
            manual_target.get("asset_class", pd.Series("", index=manual_target.index))
            .astype(str)
            .str.lower()
            .str.contains("cash", na=False)
        ],
        "market_value",
    )

    as_of = account_timestamp()
    snapshot_id = f"unified-live-{as_of.strftime('%Y%m%dT%H%M%SZ')}"
    position_records: list[dict[str, Any]] = []
    for frame in (broker_positions, manual_target):
        if not frame.empty:
            position_records.extend(frame.to_dict("records"))
    positions = tuple(_position_snapshot(row) for row in position_records)
    snapshot = AccountSnapshot(
        snapshot_id=snapshot_id,
        as_of=as_of,
        account_id=UNIFIED_LIVE_ACCOUNT_ID,
        broker=UNIFIED_LIVE_BROKER,
        profile=unified_profile,
        environment=AccountEnvironment.LIVE,
        currency=target_currency,
        net_liquidation=broker_nav_value + manual_target_value,
        cash=broker_cash_value + manual_target_cash,
        positions=positions,
        cash_balances=(
            CashSnapshot(
                currency=target_currency,
                cash=broker_cash_value + manual_target_cash,
                metadata={
                    "source": "unified_live_snapshot",
                    "broker_cash": broker_cash_value,
                    "manual_external_cash": manual_target_cash,
                },
            ),
        ),
        metadata={
            "source": "unified_live_snapshot",
            "broker_profile": broker_profile,
            "currency": target_currency,
            "broker_rows": len(broker_positions),
            "manual_rows": len(manual_positions),
            "manual_included_rows": len(manual_target),
            "excluded_manual_rows": excluded_manual_rows,
            "excluded_reason": f"manual rows without {target_currency}-ready values are not added to unified NAV",
        },
    )
    write_result = write_account_snapshot(db_path, snapshot)
    return UnifiedSnapshotResult(
        write_result=write_result,
        broker_rows=len(broker_positions),
        manual_rows=len(manual_positions),
        currency=target_currency,
        manual_usd_value=manual_target_value,
        manual_usd_cash=manual_target_cash,
        excluded_manual_rows=excluded_manual_rows,
    )


def _manual_rows_for_currency(frame: pd.DataFrame, target_currency: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    target = str(target_currency or "USD").upper()
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        source = str(row.get("currency") or "USD").upper()
        rate = _manual_conversion_rate(frame, source, target)
        if rate is None:
            continue
        converted = dict(row)
        for column in ("average_cost", "market_price", "market_value", "unrealized_pnl"):
            converted[column] = _multiply_optional(converted.get(column), rate)
        converted["currency"] = target
        metadata = _json_dict(converted.get("metadata_json"))
        metadata.update(
            {
                "valuation_converted_from": source,
                "valuation_currency": target,
                "valuation_fx_rate": rate,
            }
        )
        converted["metadata_json"] = json.dumps(metadata, sort_keys=True)
        rows.append(converted)
    return pd.DataFrame(rows, columns=frame.columns) if rows else pd.DataFrame(columns=frame.columns)


def _manual_conversion_rate(frame: pd.DataFrame, source_currency: str, target_currency: str) -> float | None:
    source = str(source_currency or "USD").upper()
    target = str(target_currency or "USD").upper()
    if source == target:
        return 1.0
    for metadata_text in frame.get("metadata_json", pd.Series(dtype=object)).dropna():
        metadata = _json_dict(metadata_text)
        native = str(metadata.get("native_currency") or "").upper()
        base = str(metadata.get("base_currency") or "").upper()
        rate = _optional_float(metadata.get("fx_rate_to_base"))
        if rate is None or rate == 0:
            continue
        if native == source and base == target:
            return rate
        if native == target and base == source:
            return 1.0 / rate
    return None


def _latest_currency(frame: pd.DataFrame) -> str | None:
    if frame.empty or "currency" not in frame:
        return None
    value = str(frame.iloc[0].get("currency") or "").upper().strip()
    return value or None


def _position_snapshot(row: dict[str, Any]) -> PositionSnapshot:
    metadata = _json_dict(row.get("metadata_json"))
    metadata.update(
        {
            "source_broker": row.get("broker"),
            "source_profile": row.get("profile"),
            "source_account_id": row.get("account_id"),
            "source_account_key": row.get("account_key"),
        }
    )
    return PositionSnapshot(
        symbol=str(row.get("symbol") or "").strip(),
        asset_class=str(row.get("asset_class") or "unknown").strip(),
        quantity=_float(row.get("quantity"), 0.0),
        average_cost=_optional_float(row.get("average_cost")),
        market_price=_optional_float(row.get("market_price")),
        market_value=_optional_float(row.get("market_value")),
        unrealized_pnl=_optional_float(row.get("unrealized_pnl")),
        currency=str(row.get("currency") or "USD").upper(),
        multiplier=_float(row.get("multiplier"), 1.0) or 1.0,
        metadata=metadata,
    )


def _latest_float(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    return _optional_float(frame.iloc[0].get(column))


def _sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    total = values.sum(min_count=1)
    return 0.0 if pd.isna(total) else float(total)


def _multiply_optional(value: Any, factor: float) -> float | None:
    parsed = _optional_float(value)
    return None if parsed is None else parsed * float(factor)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _float(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


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
