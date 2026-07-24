#!/usr/bin/env python3
"""Refresh manual external holding prices using Massive first, then yfinance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    default_account_ledger_path,
    load_manual_external_positions,
    sync_manual_external_positions_from_json,
    upsert_manual_external_positions,
)
from oqp.config import load_settings  # noqa: E402
from oqp.data import MassiveOptionsDataAdapter  # noqa: E402
from oqp.options import fetch_option_mark, fetch_option_spread_mark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=default_account_ledger_path())
    parser.add_argument("--environment", default="live")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sync_manual_external_positions_from_json(args.db)
    rows = load_manual_external_positions(args.db, environment=args.environment)
    if rows.empty:
        print("No manual external positions found.")
        return 0

    import yfinance as yf

    settings = load_settings()
    options_key = settings.massive_api_key or settings.options_api_key or settings.polygon_api_key
    options_adapter = MassiveOptionsDataAdapter(api_key=options_key) if options_key else None

    fx_cache: dict[str, float | None] = {"USD": 1.0}
    refreshed: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        item = dict(row)
        metadata = _json_dict(item.get("metadata_json"))
        item["metadata"] = metadata
        currency = str(item.get("currency") or "USD").upper()
        fx_rate = fx_cache.get(currency)
        if fx_rate is None and currency not in fx_cache:
            fx_rate = _fetch_fx_to_usd(yf, currency)
            fx_cache[currency] = fx_rate
        elif currency not in fx_cache:
            fx_rate = _fetch_fx_to_usd(yf, currency)
            fx_cache[currency] = fx_rate
        if fx_rate is not None:
            item["fx_rate_to_base"] = fx_rate

        price = _fetch_position_price(yf, item, options_adapter)
        metadata_method = None
        if isinstance(item.get("metadata"), dict):
            metadata_method = item["metadata"].get("pricing_method")
            item["metadata_json"] = json.dumps(item["metadata"], sort_keys=True)
        if price is not None:
            item["current_price"] = price
            for derived_field in (
                "local_market_value",
                "local_unrealized_pnl",
                "base_current_price",
                "base_market_value",
                "base_unrealized_pnl",
            ):
                item[derived_field] = None
            current_method = str(item.get("pricing_method") or "").lower()
            is_option = "option" in str(item.get("asset_class") or "").lower()
            if is_option and metadata_method:
                item["pricing_method"] = str(metadata_method)
            elif current_method in {"", "manual_cost", "manual_cost_fallback"}:
                item["pricing_method"] = "yfinance"
        elif item.get("pricing_method") in (None, "", "manual_cost"):
            item["pricing_method"] = "manual_cost_fallback"

        refreshed.append(item)

    output = [
        {
            "position_id": row.get("position_id"),
            "symbol": row.get("symbol"),
            "currency": row.get("currency"),
            "fx_rate_to_base": row.get("fx_rate_to_base"),
            "current_price": row.get("current_price"),
            "pricing_method": row.get("pricing_method"),
        }
        for row in refreshed
    ]
    if args.dry_run:
        print(pd.DataFrame(output).to_string(index=False))
        return 0

    written = upsert_manual_external_positions(args.db, refreshed)
    print(f"db={args.db}")
    print(f"refreshed={written}")
    print(pd.DataFrame(output).to_string(index=False))
    return 0


def _fetch_position_price(
    yf: Any,
    row: dict[str, Any],
    options_adapter: MassiveOptionsDataAdapter | None = None,
) -> float | None:
    asset_class = str(row.get("asset_class") or "").lower()
    if "cash" in asset_class:
        return 1.0
    if "option_spread" in asset_class:
        _store_underlying_spot(yf, row)
        return fetch_option_spread_mark(yf, row, options_adapter)
    if "option" in asset_class:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else _json_dict(row.get("metadata_json"))
        row["metadata"] = metadata
        _store_underlying_spot(yf, row)
        return fetch_option_mark(
            yf,
            str(row.get("quote_symbol") or row.get("underlying") or "").strip(),
            str(row.get("expiry") or "").strip(),
            str(row.get("option_type") or "call").strip().lower(),
            _float_or_none(row.get("strike")),
            options_adapter=options_adapter,
            row_metadata=metadata,
        )
    return _fetch_last_close(yf, str(row.get("quote_symbol") or row.get("symbol") or "").strip())


def _store_underlying_spot(yf: Any, row: dict[str, Any]) -> None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else _json_dict(row.get("metadata_json"))
    row["metadata"] = metadata
    symbol = str(row.get("quote_symbol") or row.get("underlying") or "").strip()
    if not symbol:
        return
    spot = _fetch_last_close(yf, symbol)
    if spot is not None:
        metadata["underlying_price"] = spot
        metadata["spot"] = spot
        metadata["spot_source"] = "yfinance"


def _fetch_last_close(yf: Any, symbol: str) -> float | None:
    if not symbol:
        return None
    try:
        history = yf.Ticker(symbol).history(period="5d")
    except Exception:
        return None
    if history is None or history.empty or "Close" not in history:
        return None
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    return None if close.empty else float(close.iloc[-1])


def _fetch_fx_to_usd(yf: Any, currency: str) -> float | None:
    currency = currency.upper()
    if currency == "USD":
        return 1.0
    return _fetch_last_close(yf, f"{currency}USD=X")


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


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


if __name__ == "__main__":
    raise SystemExit(main())
