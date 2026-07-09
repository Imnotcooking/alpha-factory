"""Cross-market monitor universe and cache-backed snapshot calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from oqp.market.cache import load_cached_market_history


@dataclass(frozen=True, slots=True)
class MarketMonitorInstrument:
    """One instrument shown in the discretionary market monitor."""

    key: str
    name: str
    symbol: str
    region: str
    asset_class: str
    category: str
    source: str = "yahoo"
    notes: str = ""

    @property
    def refresh_symbol(self) -> str | None:
        if self.source == "yahoo" and self.symbol:
            return self.symbol
        return None


DEFAULT_MARKET_MONITOR_UNIVERSE: tuple[MarketMonitorInstrument, ...] = (
    MarketMonitorInstrument("us_spy", "S&P 500 ETF", "SPY", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_qqq", "Nasdaq 100 ETF", "QQQ", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_iwm", "Russell 2000 ETF", "IWM", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_ief", "US 7-10Y Treasury ETF", "IEF", "US", "rates", "Bond ETF"),
    MarketMonitorInstrument("us_tnx", "US 10Y yield", "^TNX", "US", "rates", "Yield"),
    MarketMonitorInstrument("cn_sh50", "上证50", "000016.SS", "China", "equity", "Index"),
    MarketMonitorInstrument("cn_csi300", "沪深300", "000300.SS", "China", "equity", "Index"),
    MarketMonitorInstrument("cn_csi500", "中证500", "000905.SS", "China", "equity", "Index"),
    MarketMonitorInstrument(
        "cn_10y",
        "China 10Y government yield",
        "",
        "China",
        "rates",
        "Yield",
        source="wind_qmt_planned",
        notes="Wire from Wind/QMT once the CN data provider is active.",
    ),
    MarketMonitorInstrument("hk_hsi", "Hang Seng Index", "^HSI", "Hong Kong", "equity", "Index"),
    MarketMonitorInstrument("hk_hscei", "HS China Enterprises", "^HSCE", "Hong Kong", "equity", "Index"),
    MarketMonitorInstrument("de_dax", "DAX", "^GDAXI", "Germany", "equity", "Index"),
    MarketMonitorInstrument("jp_nikkei", "Nikkei 225", "^N225", "Japan", "equity", "Index"),
    MarketMonitorInstrument("kr_kospi", "KOSPI", "^KS11", "Korea", "equity", "Index"),
    MarketMonitorInstrument("comex_gold", "Gold futures", "GC=F", "Commodities", "commodity", "CME future"),
    MarketMonitorInstrument("wti_crude", "WTI crude futures", "CL=F", "Commodities", "commodity", "NYMEX future"),
    MarketMonitorInstrument("btc", "Bitcoin", "BTC-USD", "Crypto", "crypto", "Spot"),
    MarketMonitorInstrument("eth", "Ethereum", "ETH-USD", "Crypto", "crypto", "Spot"),
)


def market_monitor_universe() -> tuple[MarketMonitorInstrument, ...]:
    """Return the default global market monitor universe."""

    return DEFAULT_MARKET_MONITOR_UNIVERSE


def market_monitor_refresh_symbols(
    universe: Iterable[MarketMonitorInstrument] | None = None,
) -> tuple[str, ...]:
    """Return provider symbols that can be refreshed through the existing Yahoo cache."""

    instruments = tuple(universe or DEFAULT_MARKET_MONITOR_UNIVERSE)
    symbols = [item.refresh_symbol for item in instruments if item.refresh_symbol]
    return tuple(dict.fromkeys(str(symbol) for symbol in symbols if symbol))


def market_monitor_snapshot(
    universe: Iterable[MarketMonitorInstrument] | None = None,
    *,
    history_loader: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a cache-backed cross-market snapshot without making network calls."""

    instruments = tuple(universe or DEFAULT_MARKET_MONITOR_UNIVERSE)
    loader = history_loader or (lambda symbol: load_cached_market_history(symbol, lookback_days=420))
    rows: list[dict[str, object]] = []
    for item in instruments:
        if not item.refresh_symbol:
            rows.append(_planned_row(item))
            continue
        try:
            history = loader(item.refresh_symbol)
        except Exception as exc:  # pragma: no cover - defensive dashboard boundary.
            rows.append(_missing_row(item, f"cache read error: {exc}"))
            continue
        if history.empty or "Close" not in history.columns:
            rows.append(_missing_row(item, "no cached close history"))
            continue
        rows.append(_snapshot_row(item, history))
    return pd.DataFrame(rows)


def _planned_row(item: MarketMonitorInstrument) -> dict[str, object]:
    return {
        "Region": item.region,
        "Asset Class": item.asset_class,
        "Category": item.category,
        "Instrument": item.name,
        "Symbol": item.symbol or item.key,
        "Last": np.nan,
        "1D %": np.nan,
        "5D %": np.nan,
        "20D %": np.nan,
        "60D %": np.nan,
        "From 52W High": np.nan,
        "HV 20D": np.nan,
        "As Of": "",
        "Source": item.source,
        "Status": "planned",
        "Notes": item.notes,
    }


def _missing_row(item: MarketMonitorInstrument, detail: str) -> dict[str, object]:
    row = _planned_row(item)
    row["Source"] = item.source
    row["Status"] = "missing"
    row["Notes"] = detail
    return row


def _snapshot_row(item: MarketMonitorInstrument, history: pd.DataFrame) -> dict[str, object]:
    frame = history.copy()
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        return _missing_row(item, "cached history has no numeric closes")

    last = float(close.iloc[-1])
    high_52w = float(close.tail(252).max()) if len(close) else np.nan
    as_of = close.index[-1]
    return {
        "Region": item.region,
        "Asset Class": item.asset_class,
        "Category": item.category,
        "Instrument": item.name,
        "Symbol": item.symbol,
        "Last": last,
        "1D %": _pct_change(close, 1),
        "5D %": _pct_change(close, 5),
        "20D %": _pct_change(close, 20),
        "60D %": _pct_change(close, 60),
        "From 52W High": (last / high_52w) - 1.0 if high_52w else np.nan,
        "HV 20D": _annualized_hv(close, 20),
        "As Of": getattr(as_of, "date", lambda: as_of)(),
        "Source": item.source,
        "Status": "ok",
        "Notes": item.notes,
    }


def _pct_change(close: pd.Series, periods: int) -> float:
    numeric = pd.to_numeric(close, errors="coerce").dropna()
    if len(numeric) <= periods:
        return float("nan")
    base = float(numeric.iloc[-periods - 1])
    if base == 0:
        return float("nan")
    return float((float(numeric.iloc[-1]) / base) - 1.0)


def _annualized_hv(close: pd.Series, window: int) -> float:
    numeric = pd.to_numeric(close, errors="coerce").dropna()
    if len(numeric) <= window:
        return float("nan")
    returns = np.log(numeric / numeric.shift(1)).dropna()
    value = returns.tail(window).std() * np.sqrt(252)
    return float(value) if pd.notna(value) else float("nan")
