"""Conservative option premium and margin helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from oqp.options.contracts import OptionRight, normalize_option_right
from oqp.options.lifecycle import intrinsic_value


@dataclass(frozen=True, slots=True)
class OptionMarginPolicy:
    commission_per_contract: float = 0.65
    disallow_naked_short: bool = True
    naked_short_underlying_pct: float = 0.20
    naked_short_floor_pct: float = 0.10


def premium_cashflow(
    *,
    quantity: float,
    price: float,
    multiplier: float,
    commission_per_contract: float = 0.0,
) -> float:
    gross = -float(quantity) * float(price) * float(multiplier)
    commission = abs(float(quantity)) * float(commission_per_contract)
    return gross - commission


def required_margin_for_leg(
    row: pd.Series | dict[str, Any],
    *,
    quantity: float,
    price: float,
    underlying_price: float | None = None,
    policy: OptionMarginPolicy | None = None,
) -> float:
    policy = policy or OptionMarginPolicy()
    multiplier = float(row.get("multiplier") or 100.0)
    if quantity >= 0:
        return max(quantity * price * multiplier, 0.0)
    if policy.disallow_naked_short:
        raise ValueError("Naked short options are disabled by margin policy.")
    spot = float(underlying_price or row.get("underlying_price") or 0.0)
    strike = float(row.get("strike") or 0.0)
    right = normalize_option_right(row.get("right") or row.get("option_type"))
    otm = 0.0
    if spot > 0 and strike > 0:
        otm = max(strike - spot, 0.0) if right == OptionRight.CALL else max(spot - strike, 0.0)
    premium = price * multiplier
    exposure = max(spot, strike) * multiplier
    margin = premium + max(
        policy.naked_short_underlying_pct * exposure - otm * multiplier,
        policy.naked_short_floor_pct * exposure,
    )
    return abs(quantity) * max(margin, 0.0)


def expiry_pnl(
    row: pd.Series | dict[str, Any],
    *,
    quantity: float,
    entry_price: float,
    underlying_price: float | None,
) -> float:
    multiplier = float(row.get("multiplier") or 100.0)
    settlement = intrinsic_value(underlying_price, float(row.get("strike") or 0.0), row.get("right"))
    return float(quantity) * (settlement - float(entry_price)) * multiplier
