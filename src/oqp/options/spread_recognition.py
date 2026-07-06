"""Option leg grouping and spread recognition for portfolio reporting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


OCC_PATTERN = re.compile(r"^([A-Z]{1,6})\s*(\d{6})([CP])(\d{8})$")
SPACED_PATTERN = re.compile(
    r"^([A-Z][A-Z0-9. -]*)\s+(\d{4}-?\d{2}-?\d{2})\s+([CP]|CALL|PUT)\s+(\d+(?:\.\d+)?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ParsedOptionContract:
    underlying: str
    expiry: str | None
    option_type: str | None
    strike: float | None


def parse_option_contract(row: pd.Series | dict[str, Any]) -> ParsedOptionContract | None:
    metadata = _metadata(row.get("metadata_json") if hasattr(row, "get") else None)
    symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
    asset_class = str(row.get("asset_class") or row.get("asset_type") or "").lower()
    if "option" not in asset_class and not _looks_like_option(symbol, metadata):
        return None

    underlying = _first_text(
        metadata,
        "underlying",
        "underlying_symbol",
        "root",
        default="",
    )
    expiry = _first_text(metadata, "expiry", "expiration", "expiration_date", default="")
    right = _first_text(metadata, "right", "option_type", "put_call", default="")
    strike = _first_number(metadata, "strike", "strike_price")

    if not underlying or not expiry or not right or strike is None:
        parsed = _parse_symbol(symbol)
        if parsed is not None:
            underlying = underlying or parsed.underlying
            expiry = expiry or parsed.expiry or ""
            right = right or parsed.option_type or ""
            strike = strike if strike is not None else parsed.strike

    underlying = (underlying or symbol.split()[0] if symbol else underlying).upper().strip()
    option_type = _normalize_right(right)
    return ParsedOptionContract(
        underlying=underlying,
        expiry=_normalize_expiry(expiry),
        option_type=option_type,
        strike=strike,
    )


def option_leg_report(positions: pd.DataFrame, volatility: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "Spread Group",
        "Symbol",
        "Underlying",
        "Expiry",
        "DTE",
        "Type",
        "Strike",
        "Quantity",
        "Market Price",
        "Market Value",
        "Unrealized P&L",
        "Delta",
        "Gamma",
        "Theta",
        "Vega",
        "HV 5D",
        "HV 20D",
    ]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    vol_lookup = _vol_lookup(volatility)
    rows = []
    for _, raw in positions.iterrows():
        metadata = _metadata(raw.get("metadata_json"))
        try:
            from oqp.options.payoff import extract_portfolio_option_legs

            extracted_legs = extract_portfolio_option_legs(raw)
        except ImportError:
            extracted_legs = []
        if extracted_legs:
            first_leg = extracted_legs[0]
            group_contract = ParsedOptionContract(
                first_leg.underlying,
                first_leg.expiry,
                first_leg.option_type,
                first_leg.strike,
            )
            group_id = _spread_group_id(raw, group_contract, metadata)
            for leg in extracted_legs:
                current_value = leg.current_value
                entry_value = leg.entry_value
                rows.append(
                    {
                        "Spread Group": group_id,
                        "Symbol": raw.get("symbol"),
                        "Underlying": _canonical_underlying(leg.underlying),
                        "Expiry": leg.expiry,
                        "DTE": _dte(leg.expiry),
                        "Type": leg.option_type,
                        "Strike": leg.strike,
                        "Quantity": leg.quantity,
                        "Market Price": leg.current_price,
                        "Market Value": current_value,
                        "Unrealized P&L": None if current_value is None else current_value - entry_value,
                        "Delta": _first_number(leg.metadata, "delta", "Delta"),
                        "Gamma": _first_number(leg.metadata, "gamma", "Gamma"),
                        "Theta": _first_number(leg.metadata, "theta", "Theta"),
                        "Vega": _first_number(leg.metadata, "vega", "Vega"),
                        "HV 5D": vol_lookup.get(_canonical_underlying(leg.underlying), {}).get("hv_5d"),
                        "HV 20D": vol_lookup.get(_canonical_underlying(leg.underlying), {}).get("hv_20d"),
                    }
                )
            continue

        parsed = parse_option_contract(raw)
        if parsed is None:
            continue
        group_id = _spread_group_id(raw, parsed, metadata)
        underlying = _canonical_underlying(parsed.underlying)
        rows.append(
            {
                "Spread Group": group_id,
                "Symbol": raw.get("symbol"),
                "Underlying": underlying,
                "Expiry": parsed.expiry,
                "DTE": _dte(parsed.expiry),
                "Type": parsed.option_type,
                "Strike": parsed.strike,
                "Quantity": _number(raw.get("quantity")),
                "Market Price": _number(raw.get("market_price")),
                "Market Value": _number(raw.get("market_value")),
                "Unrealized P&L": _number(raw.get("unrealized_pnl")),
                "Delta": _first_number(metadata, "delta"),
                "Gamma": _first_number(metadata, "gamma"),
                "Theta": _first_number(metadata, "theta"),
                "Vega": _first_number(metadata, "vega"),
                "HV 5D": vol_lookup.get(underlying, {}).get("hv_5d"),
                "HV 20D": vol_lookup.get(underlying, {}).get("hv_20d"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def recognize_option_spreads(positions: pd.DataFrame, volatility: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "Spread ID",
        "Underlying",
        "Structure",
        "Legs",
        "Expiries",
        "DTE",
        "Net Quantity",
        "Market Value",
        "Unrealized P&L",
        "Net Delta",
        "Net Gamma",
        "Net Theta",
        "Net Vega",
        "HV 5D",
        "HV 20D",
        "Status",
    ]
    legs = option_leg_report(positions, volatility)
    if legs.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for spread_id, group in legs.groupby("Spread Group", dropna=False):
        underlying = _join_unique(group["Underlying"])
        rows.append(
            {
                "Spread ID": spread_id,
                "Underlying": underlying,
                "Structure": infer_spread_structure(group),
                "Legs": int(len(group)),
                "Expiries": _join_unique(group["Expiry"]),
                "DTE": _min_number(group["DTE"]),
                "Net Quantity": _sum_number(group["Quantity"]),
                "Market Value": _sum_number(group["Market Value"]),
                "Unrealized P&L": _sum_number(group["Unrealized P&L"]),
                "Net Delta": _sum_number(group["Delta"]),
                "Net Gamma": _sum_number(group["Gamma"]),
                "Net Theta": _sum_number(group["Theta"]),
                "Net Vega": _sum_number(group["Vega"]),
                "HV 5D": _first_valid(group["HV 5D"]),
                "HV 20D": _first_valid(group["HV 20D"]),
                "Status": "recognized" if len(group) > 1 else "single leg",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["Underlying", "DTE", "Structure"],
        na_position="last",
    )


def underlying_exposure_report(
    positions: pd.DataFrame,
    volatility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = [
        "Underlying",
        "Rows",
        "Market Value",
        "Direct Market Value",
        "Option Market Value",
        "Unrealized P&L",
        "Option Contract Shares",
        "Gross Option Contract Shares",
        "Net Option Delta",
        "Delta Dollars",
        "HV 5D",
        "HV 20D",
    ]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    from oqp.options.payoff import extract_portfolio_option_legs

    vol_lookup = _vol_lookup(volatility)
    buckets: dict[str, dict[str, Any]] = {}

    def _bucket(underlying: str) -> dict[str, Any]:
        key = _canonical_underlying(underlying)
        if key not in buckets:
            buckets[key] = {
                "Underlying": key,
                "Rows": 0,
                "Market Value": 0.0,
                "Direct Market Value": 0.0,
                "Option Market Value": 0.0,
                "Unrealized P&L": 0.0,
                "Option Contract Shares": 0.0,
                "Gross Option Contract Shares": 0.0,
                "Net Option Delta": 0.0,
                "Delta Dollars": 0.0,
                "_has_delta": False,
            }
        return buckets[key]

    for _, raw in positions.iterrows():
        metadata = _metadata(raw.get("metadata_json"))
        asset_class = str(raw.get("asset_class") or raw.get("Asset Class") or "").lower()
        symbol = str(raw.get("symbol") or raw.get("Symbol") or "").upper().strip()
        market_value = _number(raw.get("market_value"))
        if market_value is None:
            market_value = _number(raw.get("Market Value")) or 0.0
        unrealized = _number(raw.get("unrealized_pnl"))
        if unrealized is None:
            unrealized = _number(raw.get("Unrealized P&L")) or 0.0

        if "option" not in asset_class:
            underlying = _first_text(metadata, "underlying", "underlying_symbol", default=symbol) or symbol
            bucket = _bucket(underlying)
            bucket["Rows"] += 1
            bucket["Market Value"] += float(market_value)
            bucket["Direct Market Value"] += float(market_value)
            bucket["Unrealized P&L"] += float(unrealized)
            continue

        legs = extract_portfolio_option_legs(raw)
        if legs:
            row_underlyings = sorted({_canonical_underlying(leg.underlying) for leg in legs if leg.underlying})
            if not row_underlyings:
                parsed = parse_option_contract(raw)
                row_underlyings = [_canonical_underlying(parsed.underlying if parsed else symbol)]
            value_share = float(market_value) / max(len(row_underlyings), 1)
            pnl_share = float(unrealized) / max(len(row_underlyings), 1)
            for underlying in row_underlyings:
                bucket = _bucket(underlying)
                bucket["Rows"] += 1
                bucket["Market Value"] += value_share
                bucket["Option Market Value"] += value_share
                bucket["Unrealized P&L"] += pnl_share

            for leg in legs:
                bucket = _bucket(leg.underlying)
                contract_shares = float(leg.quantity) * float(leg.multiplier)
                bucket["Option Contract Shares"] += contract_shares
                bucket["Gross Option Contract Shares"] += abs(contract_shares)
                delta = _first_number(leg.metadata, "delta")
                if delta is None:
                    delta = _first_number(metadata, "delta")
                if delta is None:
                    continue
                delta_units = float(delta) * contract_shares
                bucket["Net Option Delta"] += delta_units
                bucket["_has_delta"] = True
                spot = _first_number(
                    leg.metadata,
                    "underlying_price",
                    "spot",
                    "spot_price",
                    "stock_price",
                )
                if spot is None:
                    spot = _first_number(metadata, "underlying_price", "spot", "spot_price", "stock_price")
                if spot is not None:
                    bucket["Delta Dollars"] += delta_units * float(spot)
            continue

        parsed = parse_option_contract(raw)
        underlying = _canonical_underlying(
            _first_text(metadata, "underlying", "underlying_symbol", default="")
            or (parsed.underlying if parsed is not None else symbol)
        )
        quantity = _number(raw.get("quantity"))
        if quantity is None:
            quantity = _number(raw.get("Quantity")) or 0.0
        multiplier = _number(raw.get("multiplier"))
        if multiplier is None:
            multiplier = _number(raw.get("Multiplier")) or 100.0
        contract_shares = float(quantity) * float(multiplier)
        bucket = _bucket(underlying)
        bucket["Rows"] += 1
        bucket["Market Value"] += float(market_value)
        bucket["Option Market Value"] += float(market_value)
        bucket["Unrealized P&L"] += float(unrealized)
        bucket["Option Contract Shares"] += contract_shares
        bucket["Gross Option Contract Shares"] += abs(contract_shares)
        delta = _first_number(metadata, "delta")
        if delta is not None:
            delta_units = float(delta) * contract_shares
            bucket["Net Option Delta"] += delta_units
            bucket["_has_delta"] = True
            spot = _first_number(metadata, "underlying_price", "spot", "spot_price", "stock_price")
            if spot is not None:
                bucket["Delta Dollars"] += delta_units * float(spot)

    rows = []
    for key, bucket in buckets.items():
        row = {column: bucket.get(column) for column in columns}
        row["Net Option Delta"] = bucket["Net Option Delta"] if bucket.get("_has_delta") else None
        row["Delta Dollars"] = bucket["Delta Dollars"] if bucket.get("_has_delta") else None
        row["HV 5D"] = vol_lookup.get(key, {}).get("hv_5d")
        row["HV 20D"] = vol_lookup.get(key, {}).get("hv_20d")
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows, columns=columns)
    result["sort_key"] = pd.to_numeric(result["Market Value"], errors="coerce").abs()
    return (
        result.sort_values("sort_key", ascending=False)
        .drop(columns=["sort_key"])
        .reset_index(drop=True)
    )


def infer_spread_structure(legs: pd.DataFrame) -> str:
    if legs.empty:
        return "unknown"
    if len(legs) == 1:
        option_type = str(legs.iloc[0].get("Type") or "option")
        return f"single {option_type}".strip()

    expiries = set(legs["Expiry"].dropna().astype(str))
    types = set(legs["Type"].dropna().astype(str))
    strikes = sorted(set(pd.to_numeric(legs["Strike"], errors="coerce").dropna()))
    qty = pd.to_numeric(legs["Quantity"], errors="coerce").fillna(0.0)
    has_long_short = qty.gt(0).any() and qty.lt(0).any()

    if len(legs) == 4 and len(expiries) == 1 and types == {"call", "put"}:
        return "iron condor"
    if len(strikes) == 3 and len(types) == 1 and len(expiries) == 1:
        return "butterfly"
    if len(types) == 1 and len(expiries) == 1 and len(strikes) == 2 and has_long_short:
        return f"{next(iter(types))} vertical spread"
    if len(types) == 1 and len(expiries) > 1 and len(strikes) == 1 and has_long_short:
        return "calendar spread"
    if len(types) == 1 and len(expiries) > 1 and len(strikes) > 1 and has_long_short:
        return "diagonal spread"
    if types == {"call", "put"} and len(expiries) == 1 and len(strikes) == 1:
        return "straddle"
    if types == {"call", "put"} and len(expiries) == 1 and len(strikes) > 1:
        return "strangle"
    return "multi-leg option package"


def _parse_symbol(symbol: str) -> ParsedOptionContract | None:
    compact = OCC_PATTERN.match(symbol.replace(" ", ""))
    if compact:
        root, yymmdd, right, strike_raw = compact.groups()
        expiry = f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}"
        return ParsedOptionContract(root, expiry, _normalize_right(right), int(strike_raw) / 1000.0)

    spaced = SPACED_PATTERN.match(symbol)
    if spaced:
        root, expiry, right, strike = spaced.groups()
        return ParsedOptionContract(
            root.strip().upper(),
            _normalize_expiry(expiry),
            _normalize_right(right),
            float(strike),
        )
    return None


def _looks_like_option(symbol: str, metadata: dict[str, Any]) -> bool:
    return bool(
        _first_text(metadata, "right", "option_type", "put_call", default="")
        or OCC_PATTERN.match(symbol.replace(" ", ""))
        or SPACED_PATTERN.match(symbol)
    )


def _spread_group_id(row: pd.Series, parsed: ParsedOptionContract, metadata: dict[str, Any]) -> str:
    explicit = _first_text(
        metadata,
        "spread_id",
        "strategy_id",
        "proposal_id",
        "order_id",
        default="",
    )
    if explicit:
        return explicit
    expiry = parsed.expiry or "unknown_expiry"
    return f"{parsed.underlying}:{expiry}:heuristic"


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_text(metadata: dict[str, Any], *keys: str, default: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _first_number(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metadata:
            continue
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


def _normalize_right(value: str | None) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    return None


def _normalize_expiry(value: str | None) -> str | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.date().isoformat()


def _canonical_underlying(value: str | None) -> str:
    text = str(value or "").upper().strip().replace("_", ".")
    text = " ".join(text.split())
    if text in {"BRK B", "BRK-B"}:
        return "BRK.B"
    if text in {"0700", "700", "0700.HK", "700.HK", "TCEHY", "TENCENT"}:
        return "TENCENT"
    if text in {"VUAA.DE", "VUAA.L", "VUAA.MI"}:
        return "VUAA"
    if text in {"EQAC.DE", "EQAC.MI", "EQAC.L"}:
        return "EQAC"
    return text


def _dte(expiry: str | None) -> int | None:
    if not expiry:
        return None
    parsed = pd.to_datetime(expiry, errors="coerce")
    if pd.isna(parsed):
        return None
    return int((parsed.date() - date.today()).days)


def _vol_lookup(volatility: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if volatility is None or volatility.empty or "symbol" not in volatility:
        return {}
    out = volatility.copy()
    out["symbol"] = out["symbol"].map(_canonical_underlying)
    out = out.loc[out["symbol"].astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out.set_index("symbol").to_dict("index")


def _sum_number(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _min_number(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.min())


def _first_valid(series: pd.Series) -> Any:
    clean = series.dropna()
    return None if clean.empty else clean.iloc[0]


def _join_unique(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna().unique() if str(value)]
    return ", ".join(sorted(values))
