"""Broker export parsers for portfolio ingestion."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.portfolio.symbols import to_yahoo_ticker
from oqp.portfolio.snapshots import (
    PortfolioPositionSnapshot,
    position_snapshots_to_broker_position_frame,
)


GreeksProvider = Callable[[str, str], tuple[float, float]]

FUTUBULL_REQUIRED_COLUMNS = (
    "Symbol",
    "Quantity",
    "Average Cost",
    "Current price",
    "Currency",
)
TRADING212_REQUIRED_COLUMNS = ("Action", "Time")

TRADING212_TICKER_ALIASES = {
    symbol: to_yahoo_ticker(symbol)
    for symbol in ("VWCE", "VUAA", "EQQQ", "EQAC", "ASML")
}


@dataclass(frozen=True, slots=True)
class Trading212ImportResult:
    positions: pd.DataFrame
    banked_profit: float


def futubull_option_to_occ(futu_ticker: Any) -> tuple[str | None, str | None]:
    """Translate a Futubull option string into the OCC ticker format."""

    match = re.search(
        r"([A-Z]+)\s+(\d{6})\s+([\d\.]+)([CP])",
        str(futu_ticker).upper(),
    )
    if not match:
        return None, None

    underlying, date_str, strike_str, option_type = match.groups()
    strike_formatted = f"{int(float(strike_str) * 1000):08d}"
    return f"O:{underlying}{date_str}{option_type}{strike_formatted}", underlying


def parse_futubull_csv(
    file_path: str | Path | Any,
    *,
    greeks_provider: GreeksProvider | None = None,
) -> pd.DataFrame:
    """Parse a Futubull holdings CSV into the broker-position import shape."""

    raw = _read_csv(file_path)
    _require_columns(raw, FUTUBULL_REQUIRED_COLUMNS, source="Futubull")

    df = raw[list(FUTUBULL_REQUIRED_COLUMNS)].rename(
        columns={
            "Symbol": "Ticker",
            "Quantity": "Shares",
            "Average Cost": "AvgPrice",
            "Current price": "Broker_Price",
            "Currency": "Currency",
        }
    )
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[~df["Ticker"].str.contains("/", na=False)]

    positions: list[PortfolioPositionSnapshot] = []
    for row in df.itertuples(index=False):
        ticker = str(row.Ticker).strip()
        asset_type = "Option" if len(ticker) > 10 else "Equity"
        delta = 1.0
        gamma = 0.0
        if asset_type == "Option":
            occ_ticker, underlying = futubull_option_to_occ(ticker)
            if occ_ticker and underlying and greeks_provider is not None:
                delta, gamma = greeks_provider(occ_ticker, underlying)

        positions.append(
            PortfolioPositionSnapshot(
                broker="Futubull",
                ticker=ticker,
                shares=_parse_float(row.Shares),
                avg_price=_parse_float(row.AvgPrice),
                broker_price=_parse_float(row.Broker_Price),
                currency=str(row.Currency).strip() or "USD",
                asset_type=asset_type,
                multiplier=100.0 if asset_type == "Option" else 1.0,
                broker_pnl=0.0,
                delta=float(delta),
                gamma=float(gamma),
            )
        )

    return position_snapshots_to_broker_position_frame(positions)


def parse_trading212_csv(
    file_path: str | Path | Any,
    *,
    ticker_aliases: dict[str, str] | None = None,
) -> Trading212ImportResult:
    """Reconstruct open Trading212 positions from a history CSV."""

    raw = _read_csv(file_path)
    _require_columns(raw, TRADING212_REQUIRED_COLUMNS, source="Trading212")
    aliases = ticker_aliases or TRADING212_TICKER_ALIASES

    df = raw.copy()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.sort_values("Time")

    portfolio: dict[str, dict[str, float | str]] = {}
    for row in df.to_dict("records"):
        action = str(row.get("Action", "")).lower()
        if "buy" not in action and "sell" not in action:
            continue

        raw_ticker = str(row.get("Ticker", "")).strip()
        if not raw_ticker or raw_ticker == "nan":
            continue

        ticker = aliases.get(raw_ticker, raw_ticker)
        shares = _parse_float(row.get("No. of shares", 0.0))
        total_spent = _parse_float(row.get("Total", 0.0))
        currency = str(row.get("Currency (Total)", "EUR")).strip()
        if not currency or currency == "nan":
            currency = "EUR"

        current = portfolio.setdefault(
            ticker,
            {"Shares": 0.0, "TotalCost": 0.0, "Currency": currency},
        )
        if "buy" in action:
            current["TotalCost"] = float(current["TotalCost"]) + total_spent
            current["Shares"] = float(current["Shares"]) + shares
        elif "sell" in action and float(current["Shares"]) > 0:
            avg_cost = float(current["TotalCost"]) / float(current["Shares"])
            remaining_shares = float(current["Shares"]) - shares
            if remaining_shares <= 1e-6:
                current["Shares"] = 0.0
                current["TotalCost"] = 0.0
            else:
                current["Shares"] = remaining_shares
                current["TotalCost"] = remaining_shares * avg_cost

    positions: list[PortfolioPositionSnapshot] = []
    for ticker, values in portfolio.items():
        shares = float(values["Shares"])
        if shares <= 1e-6:
            continue
        total_cost = float(values["TotalCost"])
        positions.append(
            PortfolioPositionSnapshot(
                broker="Trading212",
                ticker=ticker,
                shares=round(shares, 6),
                avg_price=round(total_cost / shares if shares > 0 else 0.0, 4),
                currency=str(values["Currency"]),
                asset_type="Equity",
                multiplier=1.0,
                broker_price=0.0,
                broker_pnl=0.0,
                delta=1.0,
                gamma=0.0,
            )
        )

    return Trading212ImportResult(
        positions=position_snapshots_to_broker_position_frame(positions),
        banked_profit=_trading212_banked_profit(df),
    )


def _read_csv(file_path: str | Path | Any) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], *, source: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{source} CSV is missing required columns: {', '.join(missing)}")


def _parse_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "")
    for token in ("$", "€", "£", "USD", "EUR", "GBP"):
        text = text.replace(token, "")
    text = text.strip()
    if not text:
        return default

    try:
        parsed = float(text)
    except ValueError:
        return default
    return -parsed if negative else parsed


def _trading212_banked_profit(df: pd.DataFrame) -> float:
    result = _numeric_series(df, "Result").sum()
    dividends = _numeric_series(
        df[df["Action"] == "Dividend (Dividend)"],
        "Total",
    ).sum()
    interest = _numeric_series(
        df[df["Action"] == "Interest on cash"],
        "Total",
    ).sum()
    return float(result + dividends + interest)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0.0)
