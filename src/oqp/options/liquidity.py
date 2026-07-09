"""Option liquidity filters and fill-price assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


FillSide = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class OptionLiquidityRule:
    min_volume: float = 0.0
    min_open_interest: float = 0.0
    max_spread_pct: float | None = 0.25
    allow_settlement_proxy: bool = True
    spread_penalty: float = 0.5


def bid_ask_mid(row: pd.Series | dict[str, Any]) -> float | None:
    bid = _number(row.get("bid"))
    ask = _number(row.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def mark_price(row: pd.Series | dict[str, Any]) -> float | None:
    for key in ("mark", "mid", "close", "last"):
        value = _number(row.get(key))
        if value is not None and value > 0:
            return value
    return bid_ask_mid(row)


def spread_pct(row: pd.Series | dict[str, Any]) -> float | None:
    bid = _number(row.get("bid"))
    ask = _number(row.get("ask"))
    mid = bid_ask_mid(row)
    if bid is None or ask is None or mid is None or mid <= 0:
        return None
    return max(ask - bid, 0.0) / mid


def passes_liquidity(row: pd.Series | dict[str, Any], rule: OptionLiquidityRule | None = None) -> bool:
    rule = rule or OptionLiquidityRule()
    volume = _number(row.get("volume")) or 0.0
    open_interest = _number(row.get("open_interest")) or 0.0
    if volume < rule.min_volume:
        return False
    if open_interest < rule.min_open_interest:
        return False
    spread = spread_pct(row)
    if spread is None:
        return bool(rule.allow_settlement_proxy and mark_price(row) is not None)
    if rule.max_spread_pct is not None and spread > rule.max_spread_pct:
        return False
    return mark_price(row) is not None


def fill_price(
    row: pd.Series | dict[str, Any],
    side: FillSide,
    rule: OptionLiquidityRule | None = None,
) -> float:
    rule = rule or OptionLiquidityRule()
    bid = _number(row.get("bid"))
    ask = _number(row.get("ask"))
    mid = bid_ask_mid(row)
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        if rule.spread_penalty >= 1.0:
            return ask if side == "buy" else bid
        penalty = max(ask - bid, 0.0) * max(rule.spread_penalty, 0.0)
        if mid is None:
            mid = (bid + ask) / 2.0
        return mid + penalty / 2.0 if side == "buy" else max(mid - penalty / 2.0, 0.0)
    mark = mark_price(row)
    if mark is None or mark <= 0:
        raise ValueError("Cannot determine option fill price without bid/ask or mark proxy.")
    return mark


def liquidity_label(row: pd.Series | dict[str, Any], rule: OptionLiquidityRule | None = None) -> str:
    rule = rule or OptionLiquidityRule()
    if not passes_liquidity(row, rule):
        return "blocked"
    if bid_ask_mid(row) is None:
        return "settlement_proxy"
    return "tradable_quote"


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed
