"""Live option-book diagnostics for portfolio reporting.

This module turns account position rows into trader-facing option diagnostics:
signed Greeks, moneyness, intrinsic/extrinsic value, IV/HV premium, and model
data quality flags. It is intentionally vanilla-first so every dashboard can
share one rigorous baseline before more advanced calibration engines are wired.
"""

from __future__ import annotations

import json
import math
from datetime import date
from typing import Any

import pandas as pd

from oqp.options.analytics import black_scholes_greeks, solve_implied_volatility
from oqp.options.payoff import PortfolioOptionLeg, extract_portfolio_option_legs


BOOK_SUMMARY_COLUMNS = [
    "Option Rows",
    "Option Legs",
    "Underlyings",
    "Contracts",
    "Row Market Value",
    "Row Unrealized P&L",
    "Net Delta Units",
    "Delta Dollars",
    "Gamma $ 1%",
    "Theta / Day",
    "Vega / 1 vol",
    "Weighted IV",
    "Weighted HV 20D",
    "IV/HV",
    "Model Gaps",
]

POSITION_DIAGNOSTIC_COLUMNS = [
    "Package",
    "Leg",
    "Underlying",
    "Expiry",
    "DTE",
    "Type",
    "Strike",
    "Quantity",
    "Spot",
    "Moneyness",
    "Entry",
    "Mark",
    "Intrinsic",
    "Extrinsic",
    "IV",
    "HV 20D",
    "IV/HV",
    "Delta",
    "Delta Units",
    "Delta Dollars",
    "Gamma $ 1%",
    "Theta / Day",
    "Vega / 1 vol",
    "Model Source",
    "Quality Flag",
]


def option_book_summary(
    positions: pd.DataFrame,
    volatility: pd.DataFrame | None = None,
    *,
    today: date | None = None,
    rate: float = 0.045,
) -> pd.DataFrame:
    """Return one-row live option-book Greeks and volatility diagnostics."""

    diagnostics = option_position_diagnostics(positions, volatility, today=today, rate=rate)
    if positions.empty and diagnostics.empty:
        return pd.DataFrame(columns=BOOK_SUMMARY_COLUMNS)

    option_rows = _option_rows(positions)
    if option_rows.empty and diagnostics.empty:
        return pd.DataFrame(columns=BOOK_SUMMARY_COLUMNS)

    contracts = _series_sum_abs(diagnostics.get("Quantity", pd.Series(dtype=float)))
    row_market_value = _series_sum(option_rows.get("market_value", pd.Series(dtype=float)))
    row_unrealized = _series_sum(option_rows.get("unrealized_pnl", pd.Series(dtype=float)))
    weighted_iv = _weighted_average(diagnostics, "IV")
    weighted_hv = _weighted_average(diagnostics, "HV 20D")
    iv_hv = weighted_iv / weighted_hv if weighted_iv is not None and weighted_hv and weighted_hv > 0 else None
    quality_flags = diagnostics.get("Quality Flag", pd.Series(dtype=str)).astype(str)
    model_gaps = int((quality_flags.ne("ok") & quality_flags.ne("")).sum()) if not quality_flags.empty else 0

    row = {
        "Option Rows": int(len(option_rows)),
        "Option Legs": int(len(diagnostics)),
        "Underlyings": _join_unique(diagnostics.get("Underlying", pd.Series(dtype=str))),
        "Contracts": contracts,
        "Row Market Value": row_market_value,
        "Row Unrealized P&L": row_unrealized,
        "Net Delta Units": _series_sum(diagnostics.get("Delta Units", pd.Series(dtype=float))),
        "Delta Dollars": _series_sum(diagnostics.get("Delta Dollars", pd.Series(dtype=float))),
        "Gamma $ 1%": _series_sum(diagnostics.get("Gamma $ 1%", pd.Series(dtype=float))),
        "Theta / Day": _series_sum(diagnostics.get("Theta / Day", pd.Series(dtype=float))),
        "Vega / 1 vol": _series_sum(diagnostics.get("Vega / 1 vol", pd.Series(dtype=float))),
        "Weighted IV": weighted_iv,
        "Weighted HV 20D": weighted_hv,
        "IV/HV": iv_hv,
        "Model Gaps": model_gaps,
    }
    return pd.DataFrame([row], columns=BOOK_SUMMARY_COLUMNS)


def option_position_diagnostics(
    positions: pd.DataFrame,
    volatility: pd.DataFrame | None = None,
    *,
    today: date | None = None,
    rate: float = 0.045,
) -> pd.DataFrame:
    """Return leg-level diagnostics for every live option row."""

    if positions.empty:
        return pd.DataFrame(columns=POSITION_DIAGNOSTIC_COLUMNS)

    anchor = today or date.today()
    spots = _spot_lookup(positions)
    vols = _vol_lookup(volatility)
    rows: list[dict[str, Any]] = []
    for index, raw in positions.iterrows():
        legs = extract_portfolio_option_legs(raw)
        if not legs:
            continue
        data = raw.to_dict()
        parent_metadata = _metadata(data.get("metadata_json"))
        parent_metadata.update(data.get("metadata") if isinstance(data.get("metadata"), dict) else {})
        package = _package_label(data, parent_metadata, index)
        parent_source = _source_label(data, parent_metadata)
        for leg in legs:
            rows.append(
                _diagnostic_row(
                    leg,
                    package=package,
                    parent_metadata=parent_metadata,
                    parent_source=parent_source,
                    spot_lookup=spots,
                    vol_lookup=vols,
                    today=anchor,
                    rate=rate,
                )
            )
    return pd.DataFrame(rows, columns=POSITION_DIAGNOSTIC_COLUMNS)


def _diagnostic_row(
    leg: PortfolioOptionLeg,
    *,
    package: str,
    parent_metadata: dict[str, Any],
    parent_source: str,
    spot_lookup: dict[str, float],
    vol_lookup: dict[str, dict[str, Any]],
    today: date,
    rate: float,
) -> dict[str, Any]:
    metadata = dict(parent_metadata)
    metadata.update(leg.metadata if isinstance(leg.metadata, dict) else {})
    underlying = str(leg.underlying or metadata.get("underlying") or "").upper().strip()
    spot = _spot_for(underlying, metadata, spot_lookup)
    dte = _dte(leg.expiry, today=today)
    time_to_expiry = max(dte or 0, 0) / 365.0
    mark = leg.current_price
    entry = leg.entry_price
    intrinsic = _intrinsic(spot, leg.strike, leg.option_type)
    extrinsic = mark - intrinsic if mark is not None and intrinsic is not None else None
    hv_20d = _number(vol_lookup.get(underlying, {}).get("hv_20d"))

    iv = _first_number(metadata, "implied_volatility", "iv")
    iv_source = _source_label({}, metadata) or parent_source
    if (
        (iv is None or iv <= 0)
        and mark is not None
        and spot
        and time_to_expiry > 0
        and _within_bsm_price_bounds(mark, spot, leg.strike, time_to_expiry, rate, leg.option_type)
    ):
        solved = solve_implied_volatility(mark, spot, leg.strike, time_to_expiry, rate, leg.option_type)  # type: ignore[arg-type]
        if solved > 0:
            iv = solved
            iv_source = "bsm_solved_iv"

    delta = _first_number(metadata, "delta")
    gamma = _first_number(metadata, "gamma")
    theta = _first_number(metadata, "theta")
    vega = _first_number(metadata, "vega")
    model_source = iv_source or parent_source or "missing"
    if spot and iv and time_to_expiry > 0 and any(value is None for value in (delta, gamma, theta, vega)):
        fallback = black_scholes_greeks(spot, leg.strike, time_to_expiry, rate, iv, leg.option_type)  # type: ignore[arg-type]
        delta = fallback["delta"] if delta is None else delta
        gamma = fallback["gamma"] if gamma is None else gamma
        theta = fallback["theta"] if theta is None else theta
        vega = fallback["vega"] if vega is None else vega
        model_source = "bsm_fallback" if model_source in {"missing", "manual_cost", "bsm_solved_iv"} else model_source

    delta_units = _scaled(delta, leg.quantity, leg.multiplier)
    delta_dollars = delta_units * spot if delta_units is not None and spot is not None else None
    gamma_cash = (
        0.5 * gamma * leg.quantity * leg.multiplier * (0.01 * spot) ** 2
        if gamma is not None and spot is not None
        else None
    )
    theta_cash = _scaled(theta, leg.quantity, leg.multiplier)
    vega_cash = _scaled(vega, leg.quantity, leg.multiplier)
    iv_hv = iv / hv_20d if iv is not None and hv_20d and hv_20d > 0 else None

    return {
        "Package": package,
        "Leg": leg.label,
        "Underlying": underlying,
        "Expiry": leg.expiry,
        "DTE": dte,
        "Type": leg.option_type,
        "Strike": leg.strike,
        "Quantity": leg.quantity,
        "Spot": spot,
        "Moneyness": spot / leg.strike if spot is not None and leg.strike > 0 else None,
        "Entry": entry,
        "Mark": mark,
        "Intrinsic": intrinsic,
        "Extrinsic": extrinsic,
        "IV": iv,
        "HV 20D": hv_20d,
        "IV/HV": iv_hv,
        "Delta": delta,
        "Delta Units": delta_units,
        "Delta Dollars": delta_dollars,
        "Gamma $ 1%": gamma_cash,
        "Theta / Day": theta_cash,
        "Vega / 1 vol": vega_cash,
        "Model Source": model_source,
        "Quality Flag": _quality_flag(
            mark=mark,
            spot=spot,
            dte=dte,
            extrinsic=extrinsic,
            iv=iv,
            hv=hv_20d,
            delta=delta,
        ),
    }


def _option_rows(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty or "asset_class" not in positions:
        return pd.DataFrame()
    mask = positions["asset_class"].astype(str).str.lower().str.contains("option", na=False)
    return positions.loc[mask].copy()


def _spot_lookup(positions: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if positions.empty:
        return out
    for _, row in positions.iterrows():
        metadata = _metadata(row.get("metadata_json"))
        symbol = str(row.get("symbol") or metadata.get("symbol") or "").upper().strip()
        asset_class = str(row.get("asset_class") or "").lower()
        underlying = str(row.get("underlying") or metadata.get("underlying") or "").upper().strip()
        explicit_spot = _first_number(metadata, "underlying_price", "spot", "spot_price")
        if underlying and explicit_spot and "option" in asset_class:
            out.setdefault(underlying, explicit_spot)
        price = _number(row.get("market_price") or row.get("current_price"))
        if symbol and price and "option" not in asset_class:
            out[symbol] = price
    return out


def _vol_lookup(volatility: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if volatility is None or volatility.empty or "symbol" not in volatility:
        return {}
    out = volatility.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out = out.loc[out["symbol"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out.set_index("symbol").to_dict("index")


def _spot_for(underlying: str, metadata: dict[str, Any], spot_lookup: dict[str, float]) -> float | None:
    explicit = _first_number(metadata, "underlying_price", "spot", "spot_price")
    if explicit is not None and explicit > 0:
        return explicit
    return spot_lookup.get(str(underlying).upper().strip())


def _intrinsic(spot: float | None, strike: float, option_type: str) -> float | None:
    if spot is None or spot <= 0 or strike <= 0:
        return None
    if option_type == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _within_bsm_price_bounds(
    mark: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
) -> bool:
    discount_strike = strike * math.exp(-rate * time_to_expiry)
    if option_type == "call":
        lower = max(spot - discount_strike, 0.0)
        upper = spot
    else:
        lower = max(discount_strike - spot, 0.0)
        upper = discount_strike
    return lower - 1e-8 <= mark <= upper + 1e-8


def _dte(expiry: str | None, *, today: date) -> int | None:
    if not expiry:
        return None
    parsed = pd.to_datetime(expiry, errors="coerce")
    if pd.isna(parsed):
        return None
    return int((parsed.date() - today).days)


def _quality_flag(
    *,
    mark: float | None,
    spot: float | None,
    dte: int | None,
    extrinsic: float | None,
    iv: float | None,
    hv: float | None,
    delta: float | None,
) -> str:
    flags: list[str] = []
    if dte is not None and dte < 0:
        flags.append("expired")
    if spot is None:
        flags.append("spot missing")
    if mark is None:
        flags.append("mark missing")
    if extrinsic is not None and extrinsic < -0.01:
        flags.append("below intrinsic")
    if iv is None or iv <= 0:
        flags.append("iv missing")
    elif hv and hv > 0:
        iv_hv = iv / hv
        if iv_hv >= 1.5:
            flags.append("iv rich")
        elif iv_hv <= 0.7:
            flags.append("iv cheap")
    if delta is None:
        flags.append("greeks missing")
    return "ok" if not flags else ", ".join(flags[:3])


def _source_label(data: dict[str, Any], metadata: dict[str, Any]) -> str:
    return str(
        metadata.get("pricing_method")
        or metadata.get("quote_source")
        or metadata.get("source")
        or data.get("pricing_method")
        or data.get("quote_source")
        or ""
    ).strip()


def _package_label(data: dict[str, Any], metadata: dict[str, Any], index: Any) -> str:
    return str(
        metadata.get("display_symbol")
        or data.get("display_symbol")
        or data.get("symbol")
        or f"Option {index}"
    )


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_number(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _number(metadata.get(key))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _scaled(value: float | None, quantity: float, multiplier: float) -> float | None:
    if value is None:
        return None
    return float(value * quantity * multiplier)


def _series_sum(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _series_sum_abs(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).abs().sum())


def _weighted_average(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame.get("Quantity", pd.Series(1.0, index=frame.index)), errors="coerce").abs()
    valid = values.notna() & weights.gt(0)
    if not valid.any():
        return None
    return float((values[valid] * weights[valid]).sum() / weights[valid].sum())


def _join_unique(series: pd.Series) -> str:
    if series.empty:
        return ""
    values = sorted({str(value) for value in series.dropna().tolist() if str(value).strip()})
    return ", ".join(values)
