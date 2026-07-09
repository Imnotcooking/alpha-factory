"""Greek normalization helpers for option backtests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from oqp.options.analytics import black_scholes_greeks
from oqp.options.lifecycle import days_to_expiry


def fill_missing_bsm_greeks(
    frame: pd.DataFrame,
    *,
    rate: float = 0.045,
    spot_col: str = "underlying_price",
) -> pd.DataFrame:
    """Fill missing vanilla Greeks where spot, strike, expiry, and IV exist."""

    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for column in ("delta", "gamma", "theta", "vega"):
        if column not in out.columns:
            out[column] = pd.NA
    for idx, row in out.iterrows():
        if all(pd.notna(row.get(column)) for column in ("delta", "gamma", "theta", "vega")):
            continue
        spot = _number(row.get(spot_col))
        strike = _number(row.get("strike"))
        iv = _number(row.get("implied_volatility") or row.get("iv"))
        if spot is None or strike is None or iv is None:
            continue
        t = max(days_to_expiry(row.get("date"), row.get("expiry")), 0) / 365.0
        if t <= 0:
            continue
        greeks = black_scholes_greeks(spot, strike, t, rate, iv, row.get("right") or "call")
        for column in ("delta", "gamma", "theta", "vega"):
            if pd.isna(out.at[idx, column]):
                out.at[idx, column] = greeks[column]
    return out


def scaled_greek_exposure(
    *,
    greek: float | None,
    quantity: float,
    multiplier: float,
    underlying_price: float | None = None,
) -> float | None:
    if greek is None:
        return None
    exposure = float(greek) * float(quantity) * float(multiplier)
    return exposure * float(underlying_price) if underlying_price is not None else exposure


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed
