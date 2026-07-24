"""Live portfolio reporting transforms for the Ops dashboard."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from oqp.market import enrich_with_historical_volatility
from oqp.options import parse_option_contract


LIVE_HOLDINGS_COLUMNS = [
    "Symbol",
    "Broker",
    "Asset Class",
    "Quantity",
    "Market Price",
    "Market Value",
    "Average Cost",
    "Unrealized P&L",
    "Realized P&L",
    "Currency",
    "Native Currency",
    "Underlying",
    "HV 5D",
    "HV 20D",
    "CCI 20",
    "BB Width",
    "BB 6M %ile",
    "BB Z",
    "Squeeze",
    "IV",
    "IV / 20D HV",
    "DTE",
    "Delta",
    "Gamma",
    "Theta",
    "Vega",
    "Spread Group",
    "As Of",
]

INDEX_EQUITY_SYMBOLS = {
    "SPY",
    "VOO",
    "IVV",
    "VTI",
    "QQQ",
    "QQQM",
    "DIA",
    "IWM",
    "RSP",
    "VT",
    "VXUS",
    "EFA",
    "EEM",
}

CASH_EQUIVALENT_SYMBOLS = {
    "BIL",
    "BOXX",
    "CLIP",
    "CSH2",
    "GBIL",
    "ICSH",
    "JPST",
    "MINT",
    "SGOV",
    "SHV",
    "TBIL",
    "TFLO",
    "USFR",
    "XEON",
}

DEFENSIVE_EQUITY_SYMBOLS = {
    "ABBV",
    "BMY",
    "CL",
    "COST",
    "CVS",
    "CVX",
    "DUK",
    "JNJ",
    "KO",
    "LLY",
    "LMT",
    "MCD",
    "MRK",
    "NEE",
    "NOC",
    "PEP",
    "PG",
    "SO",
    "UNH",
    "WMT",
    "XOM",
}

AGGRESSIVE_EQUITY_SYMBOLS = {
    "AAOI",
    "AMAT",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "COIN",
    "INTC",
    "LITE",
    "MP",
    "MRVL",
    "MSTR",
    "MU",
    "NVDA",
    "PLTR",
    "RKLB",
    "SHOP",
    "SMCI",
    "SNOW",
    "TSLA",
    "TXN",
}

CORE_EQUITY_SYMBOLS = {
    "AAPL",
    "BRK.B",
    "BRK-B",
    "GOOG",
    "GOOGL",
    "MA",
    "META",
    "MSFT",
    "V",
}

DEFAULT_SECTOR_MAP = {
    "AAPL": "Information Technology",
    "0700.HK": "Communication Services",
    "700.HK": "Communication Services",
    "AAOI": "Information Technology",
    "ABBV": "Health Care",
    "ABT": "Health Care",
    "AMD": "Information Technology",
    "AMAT": "Information Technology",
    "AMZN": "Consumer Discretionary",
    "ARM": "Information Technology",
    "ASTS": "Communication Services",
    "AVGO": "Information Technology",
    "BRK.B": "Financials",
    "BRK-B": "Financials",
    "BRK B": "Financials",
    "COST": "Consumer Staples",
    "CVX": "Energy",
    "EQAC": "ETF / Multi-sector",
    "EQAC.MI": "ETF / Multi-sector",
    "GOOG": "Communication Services",
    "GOOGL": "Communication Services",
    "GRAB": "Industrials",
    "GS": "Financials",
    "JD": "Consumer Discretionary",
    "JNJ": "Health Care",
    "LITE": "Information Technology",
    "LMT": "Industrials",
    "META": "Communication Services",
    "MP": "Materials",
    "MRVL": "Information Technology",
    "MSFT": "Information Technology",
    "MU": "Information Technology",
    "NKE": "Consumer Discretionary",
    "NVO": "Health Care",
    "NVDA": "Information Technology",
    "NU": "Financials",
    "P": "Communication Services",
    "PLTR": "Information Technology",
    "RKLB": "Industrials",
    "SMCI": "Information Technology",
    "SBUX": "Consumer Discretionary",
    "TCEHY": "Communication Services",
    "TENCENT": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "TSM": "Information Technology",
    "VUAA": "ETF / Multi-sector",
    "VUAA.DE": "ETF / Multi-sector",
    "VWCE": "ETF / Multi-sector",
    "WMT": "Consumer Staples",
    "XOM": "Energy",
}


def enriched_live_holdings(
    positions: pd.DataFrame,
    volatility: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return current holdings enriched with volatility and option metadata."""

    if positions.empty:
        return pd.DataFrame(columns=LIVE_HOLDINGS_COLUMNS)

    out = positions.copy()
    out["metadata"] = out.get("metadata_json", pd.Series("", index=out.index)).map(_metadata)
    out["native_currency"] = [
        str(item.get("native_currency") or row.get("currency") or "USD").upper()
        for item, (_, row) in zip(out["metadata"], out.iterrows())
    ]
    out["display_broker"] = [
        str(item.get("source_broker") or row.get("broker") or "").strip()
        for item, (_, row) in zip(out["metadata"], out.iterrows())
    ]
    out["display_market_price"] = [
        _first_number(item, "local_current_price") if _first_number(item, "local_current_price") is not None else row.get("market_price")
        for item, (_, row) in zip(out["metadata"], out.iterrows())
    ]
    parsed = [parse_option_contract(row) for _, row in out.iterrows()]
    out["underlying"] = [
        (
            contract.underlying
            if contract is not None
            else str(row.get("metadata", {}).get("underlying") or row.get("symbol", "")).upper()
        )
        for contract, (_, row) in zip(parsed, out.iterrows())
    ]
    out["expiry"] = [None if contract is None else contract.expiry for contract in parsed]
    out["dte"] = out["expiry"].map(_dte)
    out["spread_group"] = [
        _spread_group(row, contract)
        for contract, (_, row) in zip(parsed, out.iterrows())
    ]

    vol_frame = pd.DataFrame() if volatility is None else volatility
    out = enrich_with_historical_volatility(out, vol_frame)
    out["iv"] = out["metadata"].map(lambda item: _first_number(item, "iv", "implied_volatility"))
    out["delta"] = out["metadata"].map(lambda item: _first_number(item, "delta"))
    out["gamma"] = out["metadata"].map(lambda item: _first_number(item, "gamma"))
    out["theta"] = out["metadata"].map(lambda item: _first_number(item, "theta"))
    out["vega"] = out["metadata"].map(lambda item: _first_number(item, "vega"))
    out["iv_hv_20d"] = _ratio(out["iv"], out["hv_20d"])

    for column in (
        "quantity",
        "market_price",
        "market_value",
        "average_cost",
        "unrealized_pnl",
        "realized_pnl",
    ):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")

    display = out.rename(
        columns={
            "symbol": "Symbol",
            "display_broker": "Broker",
            "asset_class": "Asset Class",
            "quantity": "Quantity",
            "display_market_price": "Market Price",
            "market_value": "Market Value",
            "average_cost": "Average Cost",
            "unrealized_pnl": "Unrealized P&L",
            "realized_pnl": "Realized P&L",
            "currency": "Currency",
            "native_currency": "Native Currency",
            "underlying": "Underlying",
            "hv_5d": "HV 5D",
            "hv_20d": "HV 20D",
            "cci_20d": "CCI 20",
            "bb_width_20d": "BB Width",
            "bb_width_6m_pctile": "BB 6M %ile",
            "bb_z_20d": "BB Z",
            "bb_squeeze_6m": "Squeeze",
            "iv": "IV",
            "iv_hv_20d": "IV / 20D HV",
            "dte": "DTE",
            "delta": "Delta",
            "gamma": "Gamma",
            "theta": "Theta",
            "vega": "Vega",
            "spread_group": "Spread Group",
            "as_of": "As Of",
        }
    )
    return display.reindex(columns=LIVE_HOLDINGS_COLUMNS)


def asset_sleeve_mix(
    holdings: pd.DataFrame,
    *,
    cash: float | None = None,
) -> pd.DataFrame:
    """Aggregate holdings into dashboard-friendly portfolio sleeves.

    The allocation chart uses positive exposure values so short option legs and
    other signed marks still appear as visible risk sleeves.
    """

    columns = ["Sleeve", "Exposure Value", "Weight", "Rows", "Symbols"]
    rows: list[dict[str, Any]] = []
    if cash is not None and pd.notna(cash) and abs(float(cash)) > 0:
        rows.append(
            {
                "Sleeve": "Cash",
                "Exposure Value": abs(float(cash)),
                "Rows": 1,
                "Symbols": "Cash",
            }
        )

    if not holdings.empty:
        out = holdings.copy()
        out["Exposure Value"] = pd.to_numeric(
            out.get("Market Value", 0.0),
            errors="coerce",
        ).abs()
        out["Sleeve"] = out.apply(_asset_sleeve, axis=1)
        out["Sleeve Symbol"] = out.apply(_sleeve_display_symbol, axis=1)
        grouped = (
            out.loc[out["Exposure Value"] > 0]
            .groupby("Sleeve", sort=False)
            .agg(
                **{
                    "Exposure Value": ("Exposure Value", "sum"),
                    "Rows": ("Symbol", "count"),
                    "Symbols": ("Sleeve Symbol", _compact_symbols),
                }
            )
            .reset_index()
        )
        rows.extend(grouped.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result = (
        result.groupby("Sleeve", sort=False)
        .agg(
            **{
                "Exposure Value": ("Exposure Value", "sum"),
                "Rows": ("Rows", "sum"),
                "Symbols": ("Symbols", _merge_symbol_lists),
            }
        )
        .reset_index()
    )
    total = float(result["Exposure Value"].sum())
    result["Weight"] = 0.0 if total == 0 else result["Exposure Value"] / total
    order = {
        "Cash": 0,
        "Equity (index)": 1,
        "Equity (defensive)": 2,
        "Equity (core)": 3,
        "Equity (aggressive)": 4,
        "Options": 5,
        "Fixed income": 6,
        "Commodity": 7,
        "Other": 8,
    }
    result["sort_key"] = result["Sleeve"].map(order).fillna(99)
    return (
        result.sort_values(["sort_key", "Exposure Value"], ascending=[True, False])
        .drop(columns=["sort_key"])
        .reindex(columns=columns)
        .reset_index(drop=True)
    )


def position_risk_frame(
    holdings: pd.DataFrame,
    *,
    nav: float | None = None,
    cash: float | None = None,
) -> pd.DataFrame:
    """Build row-level holdings data for exposure/risk visualizations."""

    columns = [
        "Symbol",
        "Sleeve",
        "Asset Class",
        "Market Value",
        "Exposure Value",
        "Weight",
        "Unrealized P&L",
        "Unrealized P&L %",
    ]
    rows: list[dict[str, Any]] = []
    if cash is not None and pd.notna(cash) and abs(float(cash)) > 0:
        rows.append(
            {
                "Symbol": "Cash",
                "Sleeve": "Cash",
                "Asset Class": "cash",
                "Market Value": float(cash),
                "Exposure Value": abs(float(cash)),
                "Unrealized P&L": 0.0,
                "Unrealized P&L %": 0.0,
            }
        )

    if not holdings.empty:
        out = holdings.copy()
        out["Market Value"] = pd.to_numeric(out.get("Market Value", 0.0), errors="coerce").fillna(0.0)
        out["Unrealized P&L"] = pd.to_numeric(out.get("Unrealized P&L", 0.0), errors="coerce").fillna(0.0)
        out["Exposure Value"] = out["Market Value"].abs()
        out["Sleeve"] = out.apply(_asset_sleeve, axis=1)
        cost_basis = (out["Market Value"] - out["Unrealized P&L"]).abs().replace(0, pd.NA)
        out["Unrealized P&L %"] = (out["Unrealized P&L"] / cost_basis).fillna(0.0)
        rows.extend(
            out.loc[out["Exposure Value"] > 0]
            .reindex(columns=[column for column in columns if column != "Weight"])
            .to_dict("records")
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    denominator = None
    if nav is not None and pd.notna(nav) and abs(float(nav)) > 0:
        denominator = abs(float(nav))
    if denominator is None:
        total_exposure = float(result["Exposure Value"].sum())
        denominator = total_exposure if total_exposure > 0 else None
    result["Weight"] = 0.0 if denominator is None else result["Exposure Value"] / denominator
    return (
        result.reindex(columns=columns)
        .sort_values("Exposure Value", ascending=False)
        .reset_index(drop=True)
    )


def sector_exposure_frame(
    holdings: pd.DataFrame,
    *,
    cash: float | None = None,
    sector_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate current holdings into economic sector exposure."""

    columns = ["Sector", "Exposure Value", "Weight", "Rows", "Symbols", "Unrealized P&L"]
    rows: list[dict[str, Any]] = []
    if cash is not None and pd.notna(cash) and abs(float(cash)) > 0:
        rows.append(
            {
                "Sector": "Cash",
                "Exposure Value": abs(float(cash)),
                "Rows": 1,
                "Symbols": "Cash",
                "Unrealized P&L": 0.0,
            }
        )

    if not holdings.empty:
        out = holdings.copy()
        out["Exposure Value"] = pd.to_numeric(out.get("Market Value", 0.0), errors="coerce").abs()
        out["Unrealized P&L"] = pd.to_numeric(out.get("Unrealized P&L", 0.0), errors="coerce").fillna(0.0)
        out["Sector"] = out.apply(lambda row: _sector_for_row(row, sector_lookup=sector_lookup), axis=1)
        grouped = (
            out.loc[out["Exposure Value"] > 0]
            .groupby("Sector", sort=False)
            .agg(
                **{
                    "Exposure Value": ("Exposure Value", "sum"),
                    "Rows": ("Symbol", "count"),
                    "Symbols": ("Symbol", _compact_symbols),
                    "Unrealized P&L": ("Unrealized P&L", "sum"),
                }
            )
            .reset_index()
        )
        rows.extend(grouped.to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result = (
        result.groupby("Sector", sort=False)
        .agg(
            **{
                "Exposure Value": ("Exposure Value", "sum"),
                "Rows": ("Rows", "sum"),
                "Symbols": ("Symbols", _merge_symbol_lists),
                "Unrealized P&L": ("Unrealized P&L", "sum"),
            }
        )
        .reset_index()
    )
    total = float(result["Exposure Value"].sum())
    result["Weight"] = 0.0 if total == 0 else result["Exposure Value"] / total
    order = {
        "Cash": 0,
        "ETF / Multi-sector": 1,
        "Information Technology": 2,
        "Communication Services": 3,
        "Consumer Discretionary": 4,
        "Consumer Staples": 5,
        "Health Care": 6,
        "Financials": 7,
        "Industrials": 8,
        "Energy": 9,
        "Materials": 10,
        "Utilities": 11,
        "Real Estate": 12,
        "Options": 13,
        "Unknown": 99,
    }
    result["sort_key"] = result["Sector"].map(order).fillna(90)
    return (
        result.sort_values(["sort_key", "Exposure Value"], ascending=[True, False])
        .drop(columns=["sort_key"])
        .reindex(columns=columns)
        .reset_index(drop=True)
    )


def concentration_diagnostics_frame(position_risk: pd.DataFrame) -> pd.DataFrame:
    """Return high-level concentration metrics for current live exposure."""

    columns = ["Metric", "Value", "Detail"]
    if position_risk.empty:
        return pd.DataFrame(columns=columns)

    frame = position_risk.copy()
    frame["Weight"] = pd.to_numeric(frame["Weight"], errors="coerce").fillna(0.0)
    frame["Exposure Value"] = pd.to_numeric(frame["Exposure Value"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("Weight", ascending=False).reset_index(drop=True)
    non_cash = frame.loc[frame["Symbol"].astype(str) != "Cash"].copy()
    weights = frame["Weight"].clip(lower=0)
    hhi = float((weights**2).sum())
    effective_positions = None if hhi <= 0 else 1.0 / hhi
    largest = frame.iloc[0] if not frame.empty else None
    largest_non_cash = non_cash.iloc[0] if not non_cash.empty else None
    rows = [
        {
            "Metric": "Largest exposure",
            "Value": None if largest is None else float(largest["Weight"]),
            "Detail": "" if largest is None else str(largest["Symbol"]),
        },
        {
            "Metric": "Largest non-cash",
            "Value": None if largest_non_cash is None else float(largest_non_cash["Weight"]),
            "Detail": "" if largest_non_cash is None else str(largest_non_cash["Symbol"]),
        },
        {"Metric": "Top 3 weight", "Value": float(frame["Weight"].head(3).sum()), "Detail": "largest three exposures"},
        {"Metric": "Top 5 weight", "Value": float(frame["Weight"].head(5).sum()), "Detail": "largest five exposures"},
        {"Metric": "Top 10 weight", "Value": float(frame["Weight"].head(10).sum()), "Detail": "largest ten exposures"},
        {"Metric": "HHI", "Value": hhi, "Detail": "higher means more concentrated"},
        {
            "Metric": "Effective positions",
            "Value": effective_positions,
            "Detail": "1 / sum(weight^2)",
        },
        {
            "Metric": "Position rows",
            "Value": float(len(non_cash)),
            "Detail": "excluding cash row",
        },
    ]
    return pd.DataFrame(rows, columns=columns)


def concentration_curve_frame(position_risk: pd.DataFrame) -> pd.DataFrame:
    """Return sorted cumulative exposure weights for a Pareto chart."""

    columns = ["Rank", "Symbol", "Weight", "Cumulative Weight", "Exposure Value"]
    if position_risk.empty:
        return pd.DataFrame(columns=columns)
    out = position_risk.copy()
    out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").fillna(0.0).clip(lower=0)
    out["Exposure Value"] = pd.to_numeric(out["Exposure Value"], errors="coerce").fillna(0.0)
    out = out.sort_values("Weight", ascending=False).reset_index(drop=True)
    out["Rank"] = out.index + 1
    out["Cumulative Weight"] = out["Weight"].cumsum()
    return out.reindex(columns=columns)


def currency_exposure_frame(
    holdings: pd.DataFrame,
    *,
    cash: float | None = None,
    cash_currency: str = "Cash",
) -> pd.DataFrame:
    """Aggregate exposure by position currency plus cash."""

    columns = ["Currency", "Exposure Value", "Weight", "Rows", "Symbols"]
    rows: list[dict[str, Any]] = []
    if cash is not None and pd.notna(cash) and abs(float(cash)) > 0:
        rows.append(
            {
                "Currency": cash_currency or "Cash",
                "Exposure Value": abs(float(cash)),
                "Rows": 1,
                "Symbols": "Cash",
            }
        )
    if not holdings.empty:
        out = holdings.copy()
        out["Exposure Value"] = pd.to_numeric(out.get("Market Value", 0.0), errors="coerce").abs()
        out["Currency"] = out.get("Currency", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
        out.loc[out["Currency"].eq(""), "Currency"] = "Unknown"
        grouped = (
            out.loc[out["Exposure Value"] > 0]
            .groupby("Currency", sort=False)
            .agg(
                **{
                    "Exposure Value": ("Exposure Value", "sum"),
                    "Rows": ("Symbol", "count"),
                    "Symbols": ("Symbol", _compact_symbols),
                }
            )
            .reset_index()
        )
        rows.extend(grouped.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=columns)
    result = (
        pd.DataFrame(rows)
        .groupby("Currency", sort=False)
        .agg(
            **{
                "Exposure Value": ("Exposure Value", "sum"),
                "Rows": ("Rows", "sum"),
                "Symbols": ("Symbols", _merge_symbol_lists),
            }
        )
        .reset_index()
    )
    total = float(result["Exposure Value"].sum())
    result["Weight"] = 0.0 if total == 0 else result["Exposure Value"] / total
    return (
        result.sort_values("Exposure Value", ascending=False)
        .reindex(columns=columns)
        .reset_index(drop=True)
    )


def top_unrealized_positions(holdings: pd.DataFrame, *, limit: int = 10) -> pd.DataFrame:
    if holdings.empty or "Unrealized P&L" not in holdings:
        return pd.DataFrame(columns=LIVE_HOLDINGS_COLUMNS)
    out = holdings.copy()
    out["abs_unrealized"] = pd.to_numeric(out["Unrealized P&L"], errors="coerce").abs()
    return (
        out.sort_values("abs_unrealized", ascending=False)
        .head(max(int(limit), 1))
        .drop(columns=["abs_unrealized"])
    )


def _asset_sleeve(row: pd.Series) -> str:
    symbol = str(row.get("Symbol") or "").upper().strip()
    asset_class = str(row.get("Asset Class") or "").lower()
    dte = row.get("DTE")
    if "option" in asset_class or pd.notna(dte):
        return "Options"
    if (
        "cash" in asset_class
        or symbol in {"CASH", "USD", "EUR", "GBP", "HKD", "CNH"}
        or symbol in CASH_EQUIVALENT_SYMBOLS
    ):
        return "Cash"
    if any(token in asset_class for token in ("bond", "bill", "treasury", "fixed")):
        return "Fixed income"
    if any(token in asset_class for token in ("commodity", "future", "metal", "energy")):
        return "Commodity"
    if symbol in INDEX_EQUITY_SYMBOLS:
        return "Equity (index)"
    if symbol in DEFENSIVE_EQUITY_SYMBOLS:
        return "Equity (defensive)"
    if symbol in CORE_EQUITY_SYMBOLS:
        return "Equity (core)"
    if symbol in AGGRESSIVE_EQUITY_SYMBOLS:
        return "Equity (aggressive)"
    if "etf" in asset_class:
        return "Equity (index)"
    if "equity" in asset_class or "stock" in asset_class:
        return "Equity (core)"
    return "Other"


def _sleeve_display_symbol(row: pd.Series) -> str:
    """Use the underlying ticker instead of a full option contract label."""

    symbol = str(row.get("Symbol") or "").upper().strip()
    asset_class = str(row.get("Asset Class") or "").lower()
    dte = row.get("DTE")
    if "option" not in asset_class and pd.isna(dte):
        return symbol
    underlying = str(row.get("Underlying") or "").upper().strip()
    if underlying:
        return underlying
    parsed = parse_option_contract(
        {"symbol": symbol, "asset_class": asset_class or "option", "metadata_json": "{}"}
    )
    return parsed.underlying if parsed is not None and parsed.underlying else symbol


def _sector_for_row(
    row: pd.Series,
    *,
    sector_lookup: dict[str, str] | None = None,
) -> str:
    symbol = str(row.get("Symbol") or "").upper().strip()
    underlying = str(row.get("Underlying") or "").upper().strip()
    asset_class = str(row.get("Asset Class") or "").lower()
    dte = row.get("DTE")
    lookup = {
        _normalized_symbol_key(key): value
        for key, value in {**DEFAULT_SECTOR_MAP, **(sector_lookup or {})}.items()
    }
    key = underlying if ("option" in asset_class or pd.notna(dte)) and underlying else symbol
    normalized_symbol = _normalized_symbol_key(symbol)
    normalized_key = _normalized_symbol_key(key)
    if "cash" in asset_class or normalized_symbol in {"CASH", "USD", "EUR", "GBP", "HKD", "CNH"} or normalized_symbol in CASH_EQUIVALENT_SYMBOLS:
        return "Cash"
    if "option" in asset_class or pd.notna(dte):
        return lookup.get(normalized_key, "Unknown")
    if any(token in asset_class for token in ("bond", "bill", "treasury", "fixed")):
        return "Fixed income"
    if any(token in asset_class for token in ("commodity", "future", "metal", "energy")):
        return "Commodity"
    if normalized_symbol in INDEX_EQUITY_SYMBOLS or "etf" in asset_class:
        return lookup.get(normalized_symbol, "ETF / Multi-sector")
    return lookup.get(normalized_symbol, "Unknown")


def _normalized_symbol_key(symbol: str) -> str:
    text = str(symbol or "").upper().strip().replace("_", ".")
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


def _compact_symbols(values: pd.Series, *, limit: int = 8) -> str:
    symbols = [str(value).upper() for value in values.dropna().tolist() if str(value).strip()]
    unique = list(dict.fromkeys(symbols))
    if not unique:
        return ""
    shown = unique[:limit]
    suffix = "" if len(unique) <= limit else f" +{len(unique) - limit}"
    return ", ".join(shown) + suffix


def _merge_symbol_lists(values: pd.Series) -> str:
    symbols: list[str] = []
    for value in values.dropna().tolist():
        for item in str(value).replace("+", ", +").split(","):
            token = item.strip()
            if not token or token.startswith("+"):
                continue
            symbols.append(token)
    unique = list(dict.fromkeys(symbols))
    if not unique:
        return ""
    return ", ".join(unique[:8]) + ("" if len(unique) <= 8 else f" +{len(unique) - 8}")


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


def _first_number(metadata: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metadata:
            continue
        try:
            parsed = float(metadata[key])
        except (TypeError, ValueError):
            continue
        if not pd.isna(parsed):
            return parsed
    return None


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.replace(0.0, pd.NA)


def _spread_group(row: pd.Series, contract: Any) -> str | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in ("spread_id", "strategy_id", "proposal_id", "order_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    if contract is None:
        return None
    expiry = contract.expiry or "unknown_expiry"
    return f"{contract.underlying}:{expiry}:heuristic"


def _dte(expiry: str | None) -> int | None:
    if not expiry:
        return None
    parsed = pd.to_datetime(expiry, errors="coerce")
    if pd.isna(parsed):
        return None
    return int((parsed.date() - pd.Timestamp.today().date()).days)
