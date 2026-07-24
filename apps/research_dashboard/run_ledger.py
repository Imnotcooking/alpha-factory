"""Small display helpers for the research run ledger."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text.lower() not in {"", "nan", "none", "<na>", "nat"}


def _factor_name(value: Any) -> str:
    if not _present(value):
        return "Unnamed factor"
    name = str(value).strip()
    # Strategy screens append the sleeve and batch after the factor name.
    return name.split(" \u00d7 ", 1)[0].strip()


def _version(value: Any) -> str:
    if not _present(value):
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _ic(value: Any) -> str:
    if not _present(value):
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def format_run_ledger_label(row: Mapping[str, Any]) -> str:
    """Return a compact factor, market, version and IC ledger label."""

    market = row.get("market_vertical")
    if not _present(market):
        market = row.get("asset_class")
    market_label = str(market).strip() if _present(market) else "Unknown market"
    return (
        f"{_factor_name(row.get('name'))}\n"
        f"{market_label} | v{_version(row.get('round_number'))} | "
        f"IC: {_ic(row.get('holdout_ic'))}"
    )
