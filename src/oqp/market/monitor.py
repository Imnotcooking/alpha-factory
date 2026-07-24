"""Cross-market monitor universe and cache-backed snapshot calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from oqp.market.cache import (
    DEFAULT_MARKET_CACHE_MAX_AGE_HOURS,
    DEFAULT_MARKET_CACHE_PATH,
    fetch_fmp_history,
    fetch_yahoo_history,
    load_cached_market_history,
    market_cache_status,
    write_market_history,
)


@dataclass(frozen=True, slots=True)
class MarketMonitorInstrument:
    """One instrument shown in the discretionary market monitor."""

    key: str
    name: str
    symbol: str
    region: str
    asset_class: str
    category: str
    source: str = "fmp"
    notes: str = ""
    fmp_symbol: str | None = None
    yahoo_symbol: str | None = None
    enabled: bool = True

    @property
    def refresh_symbol(self) -> str | None:
        """Return the preferred FMP symbol for backward compatibility."""

        return self.fmp_refresh_symbol

    @property
    def cache_symbol(self) -> str:
        return str(self.symbol or self.key).upper().strip()

    @property
    def fmp_refresh_symbol(self) -> str | None:
        if self.source not in {"fmp", "yahoo"}:
            return None
        return str(self.fmp_symbol or self.symbol).upper().strip() or None

    @property
    def yahoo_refresh_symbol(self) -> str | None:
        if self.source not in {"fmp", "yahoo"}:
            return None
        return str(self.yahoo_symbol or self.symbol).upper().strip() or None


DEFAULT_MARKET_MONITOR_UNIVERSE: tuple[MarketMonitorInstrument, ...] = (
    MarketMonitorInstrument("us_spy", "S&P 500 ETF", "SPY", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_qqq", "Nasdaq 100 ETF", "QQQ", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_iwm", "Russell 2000 ETF", "IWM", "US", "equity", "Index ETF"),
    MarketMonitorInstrument("us_vix", "CBOE Volatility Index", "^VIX", "US", "volatility", "Volatility Index"),
    MarketMonitorInstrument("us_shy", "US 1-3Y Treasury ETF", "SHY", "US", "rates", "Bond ETF"),
    MarketMonitorInstrument("us_ief", "US 7-10Y Treasury ETF", "IEF", "US", "rates", "Bond ETF"),
    MarketMonitorInstrument("us_tlt", "US 20Y+ Treasury ETF", "TLT", "US", "rates", "Bond ETF"),
    MarketMonitorInstrument("us_tnx", "US 10Y yield", "^TNX", "US", "rates", "Yield"),
    MarketMonitorInstrument("us_lqd", "US Investment Grade Credit", "LQD", "US", "credit", "Credit ETF"),
    MarketMonitorInstrument("us_hyg", "US High Yield Credit", "HYG", "US", "credit", "Credit ETF"),
    MarketMonitorInstrument("usd_uup", "US Dollar Index Proxy", "UUP", "FX", "fx", "Dollar ETF"),
    MarketMonitorInstrument("cn_mchi", "MSCI China ETF", "MCHI", "China", "equity", "Index ETF"),
    MarketMonitorInstrument("cn_kweb", "China Internet ETF", "KWEB", "China", "equity", "Sector ETF"),
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
    MarketMonitorInstrument("em_eem", "Emerging Markets ETF", "EEM", "Global", "equity", "Index ETF"),
    MarketMonitorInstrument("fx_eurusd", "EUR / USD", "EURUSD", "FX", "fx", "Spot", yahoo_symbol="EURUSD=X"),
    MarketMonitorInstrument("fx_usdjpy", "USD / JPY", "USDJPY", "FX", "fx", "Spot", yahoo_symbol="JPY=X"),
    MarketMonitorInstrument("fx_usdcnh", "USD / CNH", "USDCNH", "FX", "fx", "Spot", yahoo_symbol="CNH=X"),
    MarketMonitorInstrument("comex_gold", "Gold", "GCUSD", "Commodities", "commodity", "Metal", yahoo_symbol="GC=F"),
    MarketMonitorInstrument("comex_copper", "Copper", "HGUSD", "Commodities", "commodity", "Metal", yahoo_symbol="HG=F"),
    MarketMonitorInstrument("wti_crude", "WTI crude", "CLUSD", "Commodities", "commodity", "Energy", yahoo_symbol="CL=F"),
    MarketMonitorInstrument("btc", "Bitcoin", "BTCUSD", "Crypto", "crypto", "Spot", yahoo_symbol="BTC-USD"),
    MarketMonitorInstrument("eth", "Ethereum", "ETHUSD", "Crypto", "crypto", "Spot", yahoo_symbol="ETH-USD"),
)


def market_monitor_universe() -> tuple[MarketMonitorInstrument, ...]:
    """Return the default global market monitor universe."""

    return tuple(item for item in DEFAULT_MARKET_MONITOR_UNIVERSE if item.enabled)


def market_monitor_refresh_symbols(
    universe: Iterable[MarketMonitorInstrument] | None = None,
) -> tuple[str, ...]:
    """Return canonical symbols for provider-backed monitor rows."""

    instruments = tuple(universe or market_monitor_universe())
    symbols = [item.cache_symbol for item in instruments if item.fmp_refresh_symbol or item.yahoo_refresh_symbol]
    return tuple(dict.fromkeys(str(symbol) for symbol in symbols if symbol))


def market_monitor_cache_status(
    universe: Iterable[MarketMonitorInstrument] | None = None,
    *,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    max_age_hours: float = DEFAULT_MARKET_CACHE_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Return provider-aware cache health for every monitor instrument."""

    instruments = tuple(universe or market_monitor_universe())
    columns = [
        "Key", "Instrument", "Symbol", "Region", "Asset Class", "Provider",
        "Rows", "First Date", "Last Date", "Fetched At", "Age Hours", "State", "Detail",
    ]
    rows: list[dict[str, Any]] = []
    for item in instruments:
        if not item.fmp_refresh_symbol and not item.yahoo_refresh_symbol:
            rows.append(
                {
                    "Key": item.key, "Instrument": item.name, "Symbol": item.cache_symbol,
                    "Region": item.region, "Asset Class": item.asset_class, "Provider": item.source,
                    "Rows": 0, "First Date": "", "Last Date": "", "Fetched At": "",
                    "Age Hours": None, "State": "planned", "Detail": item.notes,
                }
            )
            continue
        candidates = []
        for source in ("fmp", "yahoo"):
            status = market_cache_status(
                [item.cache_symbol], path=path, source=source, max_age_hours=max_age_hours
            ).iloc[0].to_dict()
            if status["State"] != "missing":
                candidates.append((source, status))
        if candidates:
            source, status = min(
                candidates,
                key=lambda pair: (pair[1]["State"] != "fresh", pair[0] != "fmp"),
            )
            detail = "" if source == "fmp" else "FMP unavailable; using Yahoo fallback cache."
        else:
            source = "fmp"
            status = market_cache_status(
                [item.cache_symbol], path=path, source="fmp", max_age_hours=max_age_hours
            ).iloc[0].to_dict()
            detail = "No cached FMP or Yahoo history."
        rows.append(
            {
                "Key": item.key, "Instrument": item.name, "Symbol": item.cache_symbol,
                "Region": item.region, "Asset Class": item.asset_class, "Provider": source,
                **{key: status.get(key) for key in ("Rows", "First Date", "Last Date", "Fetched At", "Age Hours", "State")},
                "Detail": detail,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def refresh_market_monitor_cache(
    universe: Iterable[MarketMonitorInstrument] | None = None,
    *,
    fmp_api_key: str | None,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
    period: str = "2y",
    keys: Iterable[str] | None = None,
    fmp_provider: Callable[[str, str, str], pd.DataFrame] | None = None,
    yahoo_provider: Callable[[str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Refresh monitor instruments FMP-first, falling back to Yahoo per row."""

    instruments = tuple(universe or market_monitor_universe())
    selected = {str(key) for key in keys} if keys is not None else None
    fmp_fetcher = fmp_provider or (lambda symbol, selected_period, key: fetch_fmp_history(symbol, selected_period, key))
    yahoo_fetcher = yahoo_provider or fetch_yahoo_history
    rows: list[dict[str, Any]] = []
    for item in instruments:
        if selected is not None and item.key not in selected:
            continue
        if not item.fmp_refresh_symbol and not item.yahoo_refresh_symbol:
            rows.append(_refresh_result(item, "planned", item.source, 0, item.notes))
            continue

        failures: list[str] = []
        if fmp_api_key and item.fmp_refresh_symbol:
            try:
                history = fmp_fetcher(item.fmp_refresh_symbol, period, fmp_api_key)
                written = write_market_history(item.cache_symbol, history, path=path, source="fmp")
                if written:
                    rows.append(_refresh_result(item, "ok", "fmp", written, ""))
                    continue
                failures.append(f"FMP {item.fmp_refresh_symbol}: empty response")
            except Exception as exc:  # pragma: no cover - defensive vendor boundary.
                failures.append(f"FMP {item.fmp_refresh_symbol}: {exc}")
        elif item.fmp_refresh_symbol:
            failures.append("FMP API key is not configured")

        if item.yahoo_refresh_symbol:
            try:
                history = yahoo_fetcher(item.yahoo_refresh_symbol, period)
                written = write_market_history(item.cache_symbol, history, path=path, source="yahoo")
                if written:
                    detail = "; ".join(failures) if failures else "FMP symbol unavailable"
                    rows.append(_refresh_result(item, "fallback", "yahoo", written, detail))
                    continue
                failures.append(f"Yahoo {item.yahoo_refresh_symbol}: empty response")
            except Exception as exc:  # pragma: no cover - defensive vendor boundary.
                failures.append(f"Yahoo {item.yahoo_refresh_symbol}: {exc}")

        rows.append(_refresh_result(item, "error", "none", 0, "; ".join(failures)))
    return pd.DataFrame(rows, columns=["Key", "Instrument", "Symbol", "Provider", "Status", "Rows", "Detail"])


def _refresh_result(
    item: MarketMonitorInstrument, status: str, provider: str, rows: int, detail: str
) -> dict[str, Any]:
    return {
        "Key": item.key, "Instrument": item.name, "Symbol": item.cache_symbol,
        "Provider": provider, "Status": status, "Rows": rows, "Detail": detail,
    }


def market_monitor_snapshot(
    universe: Iterable[MarketMonitorInstrument] | None = None,
    *,
    history_loader: Callable[[str], pd.DataFrame] | None = None,
    path: str | Path = DEFAULT_MARKET_CACHE_PATH,
) -> pd.DataFrame:
    """Build a cache-backed cross-market snapshot without making network calls."""

    instruments = tuple(universe or market_monitor_universe())
    rows: list[dict[str, object]] = []
    for item in instruments:
        if not item.fmp_refresh_symbol and not item.yahoo_refresh_symbol:
            rows.append(_planned_row(item))
            continue
        try:
            if history_loader is not None:
                history = history_loader(item.cache_symbol)
                provider = "injected"
                cache_state = "test"
                age_hours = None
            else:
                status = market_monitor_cache_status([item], path=path).iloc[0]
                provider = str(status["Provider"])
                cache_state = str(status["State"])
                age_hours = status["Age Hours"]
                history = load_cached_market_history(
                    item.cache_symbol, path=path, source=provider, lookback_days=420
                ) if provider in {"fmp", "yahoo"} else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive dashboard boundary.
            rows.append(_missing_row(item, f"cache read error: {exc}"))
            continue
        if history.empty or "Close" not in history.columns:
            rows.append(_missing_row(item, "no cached close history"))
            continue
        rows.append(_snapshot_row(item, history, provider=provider, cache_state=cache_state, age_hours=age_hours))
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
        "Cache State": "planned",
        "Age Hours": np.nan,
        "Status": "planned",
        "Notes": item.notes,
    }


def _missing_row(item: MarketMonitorInstrument, detail: str) -> dict[str, object]:
    row = _planned_row(item)
    row["Source"] = item.source
    row["Cache State"] = "missing"
    row["Status"] = "missing"
    row["Notes"] = detail
    return row


def _snapshot_row(
    item: MarketMonitorInstrument,
    history: pd.DataFrame,
    *,
    provider: str,
    cache_state: str,
    age_hours: object,
) -> dict[str, object]:
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
        "Source": provider,
        "Cache State": cache_state,
        "Age Hours": age_hours,
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


MARKET_MONITOR_RETURN_HORIZONS: tuple[str, ...] = ("1D %", "5D %", "20D %", "60D %")


def market_monitor_breadth(
    snapshot: pd.DataFrame,
    *,
    horizon: str = "20D %",
) -> pd.DataFrame:
    """Summarize return breadth and leadership by asset class."""

    columns = [
        "Asset Class", "Instruments", "Valid", "Positive", "Breadth",
        "Mean Return", "Median Return", "Best Symbol", "Worst Symbol",
    ]
    if snapshot.empty or horizon not in snapshot:
        return pd.DataFrame(columns=columns)
    frame = snapshot.loc[snapshot.get("Status", pd.Series(index=snapshot.index, dtype=str)).eq("ok")].copy()
    frame["_return"] = pd.to_numeric(frame[horizon], errors="coerce")
    rows: list[dict[str, Any]] = []
    for asset_class, group in frame.groupby("Asset Class", dropna=False, sort=True):
        valid = group.dropna(subset=["_return"])
        best_idx = valid["_return"].idxmax() if not valid.empty else None
        worst_idx = valid["_return"].idxmin() if not valid.empty else None
        rows.append(
            {
                "Asset Class": str(asset_class),
                "Instruments": len(group),
                "Valid": len(valid),
                "Positive": int(valid["_return"].gt(0).sum()),
                "Breadth": float(valid["_return"].gt(0).mean()) if not valid.empty else np.nan,
                "Mean Return": float(valid["_return"].mean()) if not valid.empty else np.nan,
                "Median Return": float(valid["_return"].median()) if not valid.empty else np.nan,
                "Best Symbol": str(frame.at[best_idx, "Symbol"]) if best_idx is not None else "",
                "Worst Symbol": str(frame.at[worst_idx, "Symbol"]) if worst_idx is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def market_monitor_movers(
    snapshot: pd.DataFrame,
    *,
    horizon: str = "1D %",
    count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the strongest and weakest valid instruments for one horizon."""

    if snapshot.empty or horizon not in snapshot:
        return pd.DataFrame(), pd.DataFrame()
    frame = snapshot.loc[snapshot.get("Status", pd.Series(index=snapshot.index, dtype=str)).eq("ok")].copy()
    frame["Return"] = pd.to_numeric(frame[horizon], errors="coerce")
    frame = frame.dropna(subset=["Return"]).sort_values("Return", ascending=False)
    columns = [column for column in ("Instrument", "Symbol", "Asset Class", "Return", "Source") if column in frame]
    return frame.head(max(1, count))[columns].reset_index(drop=True), frame.tail(max(1, count)).sort_values("Return")[columns].reset_index(drop=True)


def market_monitor_regime(snapshot: pd.DataFrame) -> dict[str, Any]:
    """Build an explainable risk-on/risk-off reading from liquid proxies."""

    indexed = snapshot.copy()
    if indexed.empty or "Symbol" not in indexed:
        return _empty_regime()
    indexed = indexed.drop_duplicates("Symbol", keep="last").set_index("Symbol")

    risk_symbols = ("SPY", "QQQ", "IWM", "HYG", "EEM", "MCHI", "HGUSD", "BTCUSD")
    risk_returns = pd.to_numeric(indexed.reindex(risk_symbols).get("20D %"), errors="coerce").dropna()
    risk_breadth = float(risk_returns.gt(0).mean()) if not risk_returns.empty else np.nan
    spy = _indexed_number(indexed, "SPY", "20D %")
    hyg = _indexed_number(indexed, "HYG", "20D %")
    vix = _indexed_number(indexed, "^VIX", "20D %")
    vix_level = _indexed_number(indexed, "^VIX", "Last")

    components: list[tuple[str, int, str]] = []
    if pd.notna(risk_breadth):
        signal = 1 if risk_breadth >= 0.6 else (-1 if risk_breadth <= 0.4 else 0)
        components.append(("Risk breadth", signal, f"{risk_breadth:.0%} of risk proxies positive over 20D"))
    if pd.notna(spy):
        components.append(("US equity trend", 1 if spy > 0 else -1, f"SPY 20D return {spy:.1%}"))
    if pd.notna(hyg):
        components.append(("Credit trend", 1 if hyg > 0 else -1, f"HYG 20D return {hyg:.1%}"))
    if pd.notna(vix):
        components.append(("Volatility trend", 1 if vix <= 0 else -1, f"VIX 20D change {vix:.1%}"))

    score = float(np.mean([signal for _, signal, _ in components])) if components else np.nan
    label = "Risk On" if pd.notna(score) and score >= 0.35 else (
        "Risk Off" if pd.notna(score) and score <= -0.35 else "Mixed"
    )
    return {
        "Label": label,
        "Score": score,
        "Confidence": abs(score) if pd.notna(score) else np.nan,
        "Risk Breadth": risk_breadth,
        "SPY 20D": spy,
        "HYG 20D": hyg,
        "VIX 20D": vix,
        "VIX Level": vix_level,
        "Components": pd.DataFrame(components, columns=["Component", "Signal", "Evidence"]),
    }


def _empty_regime() -> dict[str, Any]:
    return {
        "Label": "Insufficient Data", "Score": np.nan, "Confidence": np.nan,
        "Risk Breadth": np.nan, "SPY 20D": np.nan, "HYG 20D": np.nan,
        "VIX 20D": np.nan, "VIX Level": np.nan,
        "Components": pd.DataFrame(columns=["Component", "Signal", "Evidence"]),
    }


def _indexed_number(frame: pd.DataFrame, symbol: str, column: str) -> float:
    if symbol not in frame.index or column not in frame:
        return float("nan")
    value = pd.to_numeric(pd.Series([frame.at[symbol, column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float("nan")


def market_monitor_correlation(
    histories: dict[str, pd.DataFrame | pd.Series],
    *,
    lookback: int = 60,
    min_observations: int = 20,
) -> pd.DataFrame:
    """Return aligned daily-return correlations for cached instrument histories."""

    returns: dict[str, pd.Series] = {}
    for symbol, history in histories.items():
        if isinstance(history, pd.DataFrame):
            if "Close" not in history:
                continue
            close = history["Close"]
        else:
            close = history
        numeric = pd.to_numeric(close, errors="coerce").dropna().sort_index()
        daily = numeric.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna().tail(lookback)
        if len(daily) >= min_observations:
            returns[str(symbol)] = daily
    if len(returns) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(returns).tail(lookback)
    return aligned.corr(min_periods=min_observations)
