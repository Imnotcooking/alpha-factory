"""Option lifecycle and expiry-settlement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from oqp.options.contracts import OptionRight, normalize_option_right


@dataclass(frozen=True, slots=True)
class OptionLifecycleEvent:
    event_type: str
    option_symbol: str
    event_date: date
    settlement_price: float
    reason: str
    metadata: dict[str, Any] | None = None


def days_to_expiry(as_of: Any, expiry: Any) -> int:
    as_of_date = pd.to_datetime(as_of).date()
    expiry_date = pd.to_datetime(expiry).date()
    return int((expiry_date - as_of_date).days)


def is_expired(as_of: Any, expiry: Any) -> bool:
    return days_to_expiry(as_of, expiry) <= 0


def intrinsic_value(
    underlying_price: float | None,
    strike: float,
    right: str | OptionRight,
) -> float:
    if underlying_price is None or underlying_price <= 0 or strike <= 0:
        return 0.0
    normalized = normalize_option_right(right)
    if normalized == OptionRight.CALL:
        return max(float(underlying_price) - float(strike), 0.0)
    return max(float(strike) - float(underlying_price), 0.0)


def expiry_settlement_value(row: pd.Series | dict[str, Any], underlying_price: float | None) -> float:
    return intrinsic_value(
        underlying_price,
        float(row.get("strike") or 0.0),
        row.get("right") or row.get("option_type"),
    )


def lifecycle_quality(row: pd.Series | dict[str, Any]) -> str:
    missing = [
        field
        for field in ("option_symbol", "underlying_symbol", "expiry", "right", "strike", "multiplier")
        if row.get(field) in (None, "")
    ]
    return "ok" if not missing else f"missing:{','.join(missing)}"
