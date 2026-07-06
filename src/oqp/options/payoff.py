"""Portfolio option payoff and risk helpers.

These utilities are intentionally model-light: they explain the current option
book using stated legs, costs, and marks. They are not a replacement for a full
pricing engine, but they give the dashboard a reproducible first pass at payoff
shape, breakevens, and risk checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from oqp.options.analytics import black_scholes_price, solve_implied_volatility
from oqp.options.spread_recognition import parse_option_contract


@dataclass(frozen=True, slots=True)
class PortfolioOptionLeg:
    label: str
    underlying: str
    expiry: str | None
    option_type: str
    strike: float
    quantity: float
    entry_price: float
    current_price: float | None
    multiplier: float
    metadata: dict[str, Any]

    @property
    def entry_value(self) -> float:
        return self.quantity * self.entry_price * self.multiplier

    @property
    def current_value(self) -> float | None:
        if self.current_price is None:
            return None
        return self.quantity * self.current_price * self.multiplier


def extract_portfolio_option_legs(row: pd.Series | dict[str, Any]) -> list[PortfolioOptionLeg]:
    """Extract signed option legs from an account-position row."""

    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    metadata = _metadata(data.get("metadata_json"))
    metadata.update(data.get("metadata") if isinstance(data.get("metadata"), dict) else {})
    parsed = parse_option_contract(data)

    asset_class = str(data.get("asset_class") or data.get("Asset Class") or "").lower()
    package_quantity = _num(data.get("quantity") or data.get("Quantity"), 1.0)
    multiplier = _num(data.get("multiplier") or data.get("Multiplier"), 100.0) or 100.0
    underlying = str(
        data.get("underlying")
        or data.get("Underlying")
        or metadata.get("underlying")
        or (parsed.underlying if parsed is not None else "")
    ).upper()
    expiry = _text(data.get("expiry") or metadata.get("expiry") or (parsed.expiry if parsed is not None else None))
    option_type = _normalize_type(
        data.get("option_type") or metadata.get("option_type") or (parsed.option_type if parsed is not None else None)
    )

    if "option_spread" in asset_class and isinstance(metadata.get("legs"), list):
        legs: list[PortfolioOptionLeg] = []
        for index, leg in enumerate(metadata["legs"], start=1):
            if not isinstance(leg, dict):
                continue
            strike = _num(leg.get("strike"))
            if strike is None:
                continue
            leg_quantity = _num(leg.get("quantity"))
            if leg_quantity is None:
                leg_quantity = -1.0 if str(leg.get("side") or "").lower().startswith("sell") else 1.0
            leg_type = _normalize_type(leg.get("option_type") or option_type)
            entry = _num(leg.get("average_cost") or leg.get("premium") or leg.get("entry_price"), 0.0) or 0.0
            current = _num(leg.get("current_price") or leg.get("mark") or leg.get("last"))
            side = "+" if leg_quantity * package_quantity >= 0 else "-"
            right = "C" if leg_type == "call" else "P"
            legs.append(
                PortfolioOptionLeg(
                    label=f"{side}{abs(leg_quantity * package_quantity):g}x {strike:g}{right}",
                    underlying=underlying,
                    expiry=_text(leg.get("expiry") or expiry),
                    option_type=leg_type,
                    strike=float(strike),
                    quantity=float(leg_quantity * package_quantity),
                    entry_price=float(entry),
                    current_price=current,
                    multiplier=multiplier,
                    metadata=leg,
                )
            )
        return legs

    strike = _num(data.get("strike") or data.get("Strike") or metadata.get("strike") or (parsed.strike if parsed is not None else None))
    if strike is None:
        return []
    entry = _num(data.get("average_cost") or data.get("Average Cost"), 0.0) or 0.0
    current = _num(data.get("market_price") or data.get("Market Price") or data.get("current_price"))
    side = -1.0 if str(data.get("side") or metadata.get("side") or "").lower().startswith("short") else 1.0
    signed_quantity = package_quantity * side
    right = "C" if option_type == "call" else "P"
    return [
        PortfolioOptionLeg(
            label=f"{'+' if signed_quantity >= 0 else '-'}{abs(signed_quantity):g}x {float(strike):g}{right}",
            underlying=underlying,
            expiry=expiry,
            option_type=option_type,
            strike=float(strike),
            quantity=float(signed_quantity),
            entry_price=float(entry),
            current_price=current,
            multiplier=multiplier,
            metadata=metadata,
        )
    ]


def option_payoff_curve(
    row: pd.Series | dict[str, Any],
    *,
    points: int = 121,
    lower: float | None = None,
    upper: float | None = None,
) -> pd.DataFrame:
    """Return expiry P&L across underlying terminal prices."""

    legs = extract_portfolio_option_legs(row)
    if not legs:
        return pd.DataFrame(columns=["Underlying Price", "Expiry P&L"])
    strikes = [leg.strike for leg in legs]
    min_strike = min(strikes)
    max_strike = max(strikes)
    lower_bound = lower if lower is not None else max(0.01, min_strike * 0.65)
    upper_bound = upper if upper is not None else max(max_strike * 1.35, min_strike * 1.35)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound * 1.5
    prices = [lower_bound + (upper_bound - lower_bound) * index / max(points - 1, 1) for index in range(points)]
    rows = [
        {
            "Underlying Price": price,
            "Expiry P&L": _portfolio_pnl_at_expiry(legs, price),
        }
        for price in prices
    ]
    return pd.DataFrame(rows)


def option_payoff_surface(
    row: pd.Series | dict[str, Any],
    *,
    price_points: int = 61,
    time_points: int = 24,
    today: date | None = None,
    rate: float = 0.045,
) -> pd.DataFrame:
    """Build a price/time P&L surface for one option package.

    When spot and IV are available, this uses a constant-IV Black-Scholes bridge
    from today to expiry. If a full model input set is missing, it falls back to
    the older linear blend between current marked P&L and expiry payoff.
    """

    legs = extract_portfolio_option_legs(row)
    if not legs:
        return pd.DataFrame(columns=["Underlying Price", "Days To Expiry", "Illustrative P&L"])
    model_surface = _option_model_surface(
        row,
        legs,
        price_points=price_points,
        time_points=time_points,
        today=today,
        rate=rate,
    )
    if not model_surface.empty:
        return model_surface

    curve = option_payoff_curve(row, points=price_points)
    dte = max(_dte(_common_expiry(legs), today=today), 0)
    current_pnl = _current_pnl(row, legs)
    days = [round(dte * index / max(time_points - 1, 1)) for index in range(time_points)]
    rows = []
    for day in days:
        expiry_weight = 1.0 if dte == 0 else 1.0 - (day / dte)
        expiry_weight = max(0.0, min(1.0, expiry_weight))
        for _, point in curve.iterrows():
            expiry_pnl = _num(point.get("Expiry P&L"), 0.0) or 0.0
            blended = current_pnl * (1.0 - expiry_weight) + expiry_pnl * expiry_weight
            rows.append(
                {
                    "Underlying Price": point["Underlying Price"],
                    "Days To Expiry": day,
                    "Illustrative P&L": blended,
                }
            )
    return pd.DataFrame(rows)


def _option_model_surface(
    row: pd.Series | dict[str, Any],
    legs: list[PortfolioOptionLeg],
    *,
    price_points: int,
    time_points: int,
    today: date | None,
    rate: float,
) -> pd.DataFrame:
    dte = max(_dte(_common_expiry(legs), today=today), 0)
    if dte <= 0:
        return pd.DataFrame()

    metadata = _metadata((row.to_dict() if isinstance(row, pd.Series) else dict(row)).get("metadata_json"))
    spot = _num(
        metadata.get("underlying_price")
        or metadata.get("spot")
        or metadata.get("spot_price")
    )
    if spot is None or spot <= 0:
        return pd.DataFrame()

    leg_models: list[tuple[PortfolioOptionLeg, float]] = []
    current_t = max(dte, 1) / 365.0
    for leg in legs:
        leg_metadata = dict(metadata)
        leg_metadata.update(leg.metadata if isinstance(leg.metadata, dict) else {})
        iv = _num(leg_metadata.get("implied_volatility") or leg_metadata.get("iv"))
        if (iv is None or iv <= 0) and leg.current_price is not None:
            solved = solve_implied_volatility(
                float(leg.current_price),
                float(spot),
                float(leg.strike),
                current_t,
                rate,
                leg.option_type,
            )
            iv = solved if solved > 0 else None
        if iv is None or iv <= 0:
            return pd.DataFrame()
        leg_models.append((leg, float(iv)))

    curve = option_payoff_curve(row, points=price_points)
    if curve.empty:
        return pd.DataFrame()
    prices = curve["Underlying Price"].astype(float).tolist()
    days = [round(dte * index / max(time_points - 1, 1)) for index in range(time_points)]
    rows = []
    for day in days:
        remaining_t = max(day, 0) / 365.0
        for price in prices:
            model_value = 0.0
            entry_value = 0.0
            for leg, iv in leg_models:
                if remaining_t <= 0:
                    option_value = max(price - leg.strike, 0.0) if leg.option_type == "call" else max(leg.strike - price, 0.0)
                else:
                    option_value = black_scholes_price(price, leg.strike, remaining_t, rate, iv, leg.option_type)
                model_value += leg.quantity * option_value * leg.multiplier
                entry_value += leg.entry_value
            rows.append(
                {
                    "Underlying Price": price,
                    "Days To Expiry": day,
                    "Illustrative P&L": model_value - entry_value,
                    "Surface Model": "constant_iv_bsm",
                }
            )
    return pd.DataFrame(rows)


def option_risk_summary(
    row: pd.Series | dict[str, Any],
    *,
    today: date | None = None,
) -> pd.DataFrame:
    """Return compact risk checkpoints for one option package."""

    legs = extract_portfolio_option_legs(row)
    columns = ["Metric", "Value", "Notes"]
    if not legs:
        return pd.DataFrame(columns=columns)

    curve = option_payoff_curve(row)
    min_pnl = float(curve["Expiry P&L"].min()) if not curve.empty else None
    max_pnl = float(curve["Expiry P&L"].max()) if not curve.empty else None
    exact = _defined_vertical_risk(legs)
    if exact:
        min_pnl = exact["max_loss"]
        max_pnl = exact["max_profit"]

    entry_debit = sum(leg.entry_value for leg in legs)
    current_pnl = _current_pnl(row, legs)
    breakevens = _breakevens(curve)
    dte = _dte(_common_expiry(legs), today=today)
    risk_unit = abs(min_pnl) if min_pnl is not None and min_pnl < 0 else abs(entry_debit)
    rows = [
        {"Metric": "Structure", "Value": " / ".join(leg.label for leg in legs), "Notes": _common_expiry(legs) or "expiry missing"},
        {"Metric": "DTE", "Value": dte, "Notes": "Calendar days to expiration."},
        {"Metric": "Entry Debit", "Value": entry_debit, "Notes": "Positive means cash paid; negative means net credit."},
        {"Metric": "Current P&L", "Value": current_pnl, "Notes": "Uses current mark if available, otherwise account row P&L."},
        {"Metric": "Max Loss", "Value": min_pnl, "Notes": "Exact for simple verticals; otherwise grid estimate."},
        {"Metric": "Max Profit", "Value": max_pnl, "Notes": "Exact for simple verticals; single long options remain grid-estimated here."},
        {"Metric": "Breakeven", "Value": ", ".join(f"{value:.2f}" for value in breakevens) or "not crossed", "Notes": "Estimated from expiry payoff curve."},
        {"Metric": "TP 50%", "Value": None if max_pnl is None else max_pnl * 0.50, "Notes": "First take-profit checkpoint."},
        {"Metric": "TP 75%", "Value": None if max_pnl is None else max_pnl * 0.75, "Notes": "Second take-profit checkpoint."},
        {"Metric": "SL 50%", "Value": -risk_unit * 0.50, "Notes": "Stop-loss checkpoint based on defined/grid max risk."},
    ]
    return pd.DataFrame(rows, columns=columns)


def option_greeks_frame(row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Return leg-level marks and Greeks when the data source supplied them."""

    legs = extract_portfolio_option_legs(row)
    columns = [
        "Leg",
        "Underlying",
        "Expiry",
        "Type",
        "Strike",
        "Quantity",
        "Mark",
        "IV",
        "Delta",
        "Gamma",
        "Theta",
        "Vega",
        "Source",
    ]
    rows = []
    for leg in legs:
        meta = leg.metadata
        rows.append(
            {
                "Leg": leg.label,
                "Underlying": leg.underlying,
                "Expiry": leg.expiry,
                "Type": leg.option_type,
                "Strike": leg.strike,
                "Quantity": leg.quantity,
                "Mark": leg.current_price,
                "IV": _num(meta.get("implied_volatility") or meta.get("iv")),
                "Delta": _num(meta.get("delta")),
                "Gamma": _num(meta.get("gamma")),
                "Theta": _num(meta.get("theta")),
                "Vega": _num(meta.get("vega")),
                "Source": meta.get("pricing_method") or meta.get("quote_source") or "missing",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _portfolio_pnl_at_expiry(legs: list[PortfolioOptionLeg], price: float) -> float:
    terminal_value = 0.0
    entry_value = 0.0
    for leg in legs:
        intrinsic = max(price - leg.strike, 0.0) if leg.option_type == "call" else max(leg.strike - price, 0.0)
        terminal_value += leg.quantity * intrinsic * leg.multiplier
        entry_value += leg.entry_value
    return terminal_value - entry_value


def _current_pnl(row: pd.Series | dict[str, Any], legs: list[PortfolioOptionLeg]) -> float:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    row_pnl = _num(data.get("unrealized_pnl") or data.get("Unrealized P&L"))
    current_values = [leg.current_value for leg in legs]
    if all(value is not None for value in current_values):
        return float(sum(current_values)) - sum(leg.entry_value for leg in legs)
    return 0.0 if row_pnl is None else float(row_pnl)


def _defined_vertical_risk(legs: list[PortfolioOptionLeg]) -> dict[str, float] | None:
    if len(legs) != 2:
        return None
    if len({leg.option_type for leg in legs}) != 1 or len({leg.expiry for leg in legs}) != 1:
        return None
    if not (legs[0].quantity * legs[1].quantity < 0):
        return None
    if abs(abs(legs[0].quantity) - abs(legs[1].quantity)) > 1e-8:
        return None
    width = abs(legs[0].strike - legs[1].strike) * abs(legs[0].quantity) * legs[0].multiplier
    debit = sum(leg.entry_value for leg in legs)
    if debit >= 0:
        return {"max_loss": -debit, "max_profit": max(width - debit, 0.0)}
    return {"max_loss": -(width + debit), "max_profit": abs(debit)}


def _breakevens(curve: pd.DataFrame) -> list[float]:
    if curve.empty:
        return []
    values = curve[["Underlying Price", "Expiry P&L"]].dropna().to_dict("records")
    out: list[float] = []
    for previous, current in zip(values, values[1:], strict=False):
        prev_pnl = float(previous["Expiry P&L"])
        curr_pnl = float(current["Expiry P&L"])
        if prev_pnl == 0:
            out.append(float(previous["Underlying Price"]))
        if prev_pnl * curr_pnl < 0:
            x0 = float(previous["Underlying Price"])
            x1 = float(current["Underlying Price"])
            out.append(x0 + (0.0 - prev_pnl) * (x1 - x0) / (curr_pnl - prev_pnl))
    deduped: list[float] = []
    for value in out:
        if not any(abs(value - existing) < 0.01 for existing in deduped):
            deduped.append(value)
    return deduped


def _common_expiry(legs: list[PortfolioOptionLeg]) -> str | None:
    expiries = {leg.expiry for leg in legs if leg.expiry}
    return next(iter(expiries)) if len(expiries) == 1 else None


def _dte(expiry: str | None, *, today: date | None = None) -> int | None:
    if not expiry:
        return None
    try:
        expiry_date = date.fromisoformat(str(expiry))
    except ValueError:
        return None
    ref = today or pd.Timestamp.utcnow().date()
    return (expiry_date - ref).days


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _normalize_type(value: Any) -> str:
    text = str(value or "call").strip().lower()
    return "put" if text.startswith("p") else "call"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _num(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(parsed) else parsed
