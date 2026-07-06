"""Portfolio ticker normalization helpers."""

from __future__ import annotations


PORTFOLIO_TICKER_ALIASES: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BRK B": "BRK-B",
    "VWCE": "VWCE.DE",
    "VUAA": "VUAA.DE",
    "EQQQ": "EQQQ.L",
    "EQAC": "EQQB.DE",
    "XEON": "XEON.DE",
}


def to_yahoo_ticker(ticker: str) -> str:
    """Map broker/export symbols to Yahoo Finance symbols used by yfinance."""

    normalized = str(ticker).strip().upper()
    return PORTFOLIO_TICKER_ALIASES.get(normalized, str(ticker).strip())
