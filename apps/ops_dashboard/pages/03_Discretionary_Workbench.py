"""Discretionary trade workbench for the unified Ops dashboard."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode
except ImportError:  # pragma: no cover - optional dashboard dependency
    AgGrid = None
    DataReturnMode = None
    GridOptionsBuilder = None
    GridUpdateMode = None
    JsCode = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.config import load_settings  # noqa: E402
from oqp.data import MassiveOptionsDataAdapter, OptionChainRequest  # noqa: E402
from oqp.domain import AssetClass, Instrument  # noqa: E402
from oqp.execution import write_option_trade_proposal_from_candidate  # noqa: E402
from oqp.investing import (  # noqa: E402
    DEFAULT_OPPORTUNITY_HISTORY_DB_PATH,
    DEFAULT_LLM_EVIDENCE_DB_PATH,
    DEFAULT_STOCK_WATCHLIST_PATH,
    DEFAULT_NEWS_NLP_DB_PATH,
    NewsNLPResult,
    RATIO_CATEGORIES,
    VALUATION_MULTIPLE_COLUMNS,
    add_stock_watchlist_symbol,
    analyze_company_outlook,
    analyze_news_articles,
    build_dcf_assumption_evidence,
    build_decision_checklist_frame,
    build_management_keyword_frame,
    build_management_tone_frame,
    build_news_catalyst_board,
    build_news_evidence_frame,
    build_options_playbook_frame,
    build_opportunity_lens_frame,
    build_thesis_draft,
    build_vehicle_route_frame,
    calculate_dcf_valuation,
    catalyst_evidence_text,
    estimate_dcf_assumptions,
    fetch_dcf_source_documents,
    fetch_fundamental_data,
    fetch_peer_comparison,
    fetch_price_target_consensus,
    format_compact_currency,
    fetch_rapidapi_earnings_transcript_bundle,
    load_or_refresh_news_articles,
    load_opportunity_history,
    load_stock_watchlist,
    management_tone_summary,
    news_cache_status,
    nlp_provider_status,
    primary_route,
    remove_stock_watchlist_symbol,
    safe_num,
    write_opportunity_snapshot,
)
from oqp.intelligence.signal_engine import (  # noqa: E402
    DirectionalLensResult,
    add_strategy_direction_columns,
    build_directional_lens,
    fetch_directional_sentiment,
)
from oqp.market import (  # noqa: E402
    DEFAULT_MARKET_CACHE_MAX_AGE_HOURS,
    DEFAULT_MARKET_CACHE_PATH,
    DEFAULT_VOL_FORECAST_DB_PATH,
    DEFAULT_VOL_FORECAST_HORIZONS,
    forecast_volatility_models,
    load_cached_market_history,
    market_cache_status,
    refresh_yahoo_market_cache,
    select_forecast_vol,
    write_volatility_forecasts,
)
from oqp.options import (  # noqa: E402
    black_scholes_price,
    choose_expiration,
    format_scanner_frame,
    scan_backspreads,
    scan_cash_secured_puts,
    scan_call_butterflies,
    scan_calendar_spreads,
    scan_iron_condors,
    scan_long_options,
    scan_ratio_spreads,
    scan_vertical_spreads,
    simulate_single_option,
    volatility_snapshot,
)
from oqp.ui import (  # noqa: E402
    apply_ops_theme,
    language_selector,
    ops_tabs,
    ops_text,
    page_header,
    render_dark_table,
    style_dark_plotly,
)


SCRATCHPAD_PATH = REPO_ROOT / "runtime" / "state" / "discretionary" / "trade_scratchpad.json"


st.set_page_config(
    page_title="Discretionary Workbench",
    layout="wide",
    page_icon="WORK",
    initial_sidebar_state="expanded",
)

OPS_LANG = language_selector()
apply_ops_theme()


def T(key: str, default: str | None = None, **format_values: Any) -> str:
    return ops_text(OPS_LANG, key, default, **format_values)


WORKBENCH_VIEWS = ["Watchlist", "Opportunity Hub", "API Status"]


def workbench_view_label(view: str) -> str:
    labels = dict(zip(WORKBENCH_VIEWS, ops_tabs(OPS_LANG, "workbench_nav_views")))
    return labels.get(view, view)


def apply_discretionary_workbench_style() -> None:
    """Page-local styling for interactive controls that must stay native."""

    st.markdown(
        """
        <style>
        div[data-testid="stSegmentedControl"] {
            margin: 0.4rem 0 1.25rem 0;
        }
        div[data-testid="stSegmentedControl"] div[role="radiogroup"] {
            gap: 0;
            width: fit-content;
            max-width: 100%;
            padding: 0.22rem;
            border: 1px solid rgba(78, 99, 129, 0.32);
            border-radius: 999px;
            background: rgba(5, 11, 19, 0.72);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025), 0 16px 34px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stSegmentedControl"] button,
        div[data-testid="stSegmentedControl"] label,
        div[data-testid="stSegmentedControl"] div[role="radio"],
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"] {
            background: rgba(12, 22, 35, 0.94) !important;
            border: 1px solid rgba(82, 103, 132, 0.22) !important;
            border-radius: 999px !important;
            color: #e5eefb !important;
            box-shadow: none !important;
            min-height: 2.15rem;
            padding: 0.35rem 0.95rem !important;
        }
        div[data-testid="stSegmentedControl"] button *,
        div[data-testid="stSegmentedControl"] label *,
        div[data-testid="stSegmentedControl"] div[role="radio"] * {
            color: #e5eefb !important;
        }
        div[data-testid="stSegmentedControl"] button[aria-checked="true"],
        div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
        div[data-testid="stSegmentedControl"] button[data-baseweb="button"][aria-selected="true"],
        div[data-testid="stSegmentedControl"] label:has(input:checked),
        div[data-testid="stSegmentedControl"] div[role="radio"][aria-checked="true"],
        div[data-testid="stSegmentedControl"] [data-baseweb="radio"]:has(input:checked) {
            color: #f8fafc !important;
            border-color: rgba(248, 113, 113, 0.78) !important;
            background:
                linear-gradient(90deg, rgba(244, 63, 94, 0.42), rgba(37, 99, 235, 0.35)),
                rgba(17, 27, 43, 0.98) !important;
        }
        div[data-testid="stSegmentedControl"] label:has(input:checked) *,
        div[data-testid="stSegmentedControl"] div[role="radio"][aria-checked="true"] * {
            color: #f8fafc !important;
        }
        div[data-testid="stDataFrame"] {
            --gdg-accent-color: #38bdf8;
            --gdg-accent-fg: #020617;
            --gdg-accent-light: rgba(45, 212, 191, 0.16);
            --gdg-text-dark: #e5eefb;
            --gdg-text-medium: #b7c4d6;
            --gdg-text-light: #7f8da3;
            --gdg-text-header: #8fa2bd;
            --gdg-text-group-header: #aab8ca;
            --gdg-text-header-selected: #f8fafc;
            --gdg-bg-cell: #07101a;
            --gdg-bg-cell-medium: #0b1522;
            --gdg-bg-header: #101927;
            --gdg-bg-header-has-focus: #162235;
            --gdg-bg-header-hovered: #172438;
            --gdg-bg-bubble: #132033;
            --gdg-bg-bubble-selected: #1f2f46;
            --gdg-bg-search-result: rgba(245, 158, 11, 0.18);
            --gdg-border-color: rgba(88, 108, 136, 0.22);
            --gdg-horizontal-border-color: rgba(88, 108, 136, 0.16);
            --gdg-link-color: #60a5fa;
            background: rgba(6, 12, 20, 0.92) !important;
            border: 1px solid rgba(65, 84, 112, 0.32);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] canvas {
            background: #07101a !important;
        }
        div[data-testid="stDataFrame"] input,
        div[data-testid="stDataFrame"] textarea {
            background: #0b1522 !important;
            color: #e5eefb !important;
        }
        .oqp-watchlist-table-wrap {
            max-height: 560px;
            overflow: auto;
            border: 1px solid rgba(72, 92, 122, 0.30);
            border-radius: 10px;
            background: rgba(5, 11, 19, 0.92);
            box-shadow: 0 24px 55px rgba(0, 0, 0, 0.20);
        }
        .oqp-watchlist-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            color: #dbeafe;
            font-size: 0.84rem;
            line-height: 1.38;
        }
        .oqp-watchlist-table thead th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #0f1826;
            color: #8fa2bd;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            text-align: right;
            border-bottom: 1px solid rgba(93, 113, 143, 0.30);
            padding: 0.75rem 0.78rem;
            white-space: nowrap;
        }
        .oqp-watchlist-table thead th:first-child,
        .oqp-watchlist-table thead th:nth-child(2) {
            text-align: left;
        }
        .oqp-watchlist-table thead a {
            color: #9db0ca;
            text-decoration: none;
        }
        .oqp-watchlist-table thead a:hover {
            color: #f8fafc;
        }
        .oqp-watchlist-table thead a.active-sort {
            color: #f8fafc;
        }
        .oqp-watchlist-table tbody td {
            background: rgba(8, 15, 25, 0.82);
            border-bottom: 1px solid rgba(65, 84, 112, 0.16);
            color: #dbeafe;
            padding: 0.70rem 0.78rem;
            text-align: right;
            white-space: nowrap;
        }
        .oqp-watchlist-table tbody tr:nth-child(even) td {
            background: rgba(12, 21, 33, 0.82);
        }
        .oqp-watchlist-table tbody tr:hover td {
            background: rgba(21, 35, 54, 0.96);
        }
        .oqp-watchlist-table tbody td:first-child,
        .oqp-watchlist-table tbody td:nth-child(2) {
            text-align: left;
        }
        .oqp-watchlist-table .symbol-link {
            color: #60a5fa;
            text-decoration: none;
            font-weight: 800;
        }
        .oqp-watchlist-table .symbol-link:hover {
            color: #93c5fd;
            text-decoration: underline;
        }
        .oqp-watchlist-table .heart-link {
            color: #fb7185;
            text-decoration: none;
            font-size: 1.05rem;
            font-weight: 900;
        }
        .oqp-watchlist-table .heart-link:hover {
            color: #fecdd3;
        }
        .oqp-watchlist-table .opportunity-cell {
            color: #cbd5e1;
            font-weight: 700;
            text-align: left;
        }
        .oqp-workbench-nav {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem;
            margin: 0.35rem 0 1.25rem 0;
            border: 1px solid rgba(78, 99, 129, 0.34);
            border-radius: 999px;
            background: rgba(5, 11, 19, 0.82);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03), 0 16px 34px rgba(0, 0, 0, 0.24);
        }
        .oqp-workbench-nav a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2.1rem;
            padding: 0.38rem 1.05rem;
            border: 1px solid rgba(82, 103, 132, 0.24);
            border-radius: 999px;
            background: rgba(12, 22, 35, 0.92);
            color: #dbeafe;
            font-weight: 800;
            text-decoration: none;
        }
        .oqp-workbench-nav a:hover {
            color: #f8fafc;
            border-color: rgba(96, 165, 250, 0.55);
            background: rgba(18, 32, 50, 0.98);
        }
        .oqp-workbench-nav a.is-active {
            color: #f8fafc;
            border-color: rgba(248, 113, 113, 0.82);
            background:
                linear-gradient(90deg, rgba(244, 63, 94, 0.45), rgba(37, 99, 235, 0.36)),
                rgba(17, 27, 43, 0.98);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_discretionary_workbench_style()


def apply_workbench_query_params() -> None:
    """Let watchlist links deep-link into the Opportunity Hub."""

    hub_symbol = str(st.query_params.get("hub_symbol") or "").upper().strip()
    requested_view = str(st.query_params.get("workbench_view") or "").strip()
    if hub_symbol:
        st.session_state["hub_symbol"] = hub_symbol
        st.session_state["workbench_symbol"] = hub_symbol
        st.session_state["workbench_view"] = "Opportunity Hub"
        try:
            st.query_params.clear()
        except Exception:
            pass
    elif requested_view in WORKBENCH_VIEWS:
        st.session_state["workbench_view"] = requested_view


def progress_bar(label: str, *, estimate_seconds: int) -> tuple[Any, Any]:
    progress = st.progress(0, text=f"{label} | estimated wait: ~{estimate_seconds}s")
    status = st.empty()
    return progress, status


def progress_step(progress: Any, status: Any, value: int, message: str) -> None:
    progress.progress(min(max(int(value), 0), 100), text=message)
    status.caption(message)


def finish_progress(progress: Any, status: Any) -> None:
    progress.empty()
    status.empty()


apply_workbench_query_params()


def money(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "missing"
    if pd.isna(parsed):
        return "missing"
    return f"${parsed:,.2f}"


def signed_money(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "missing"
    if pd.isna(parsed):
        return "missing"
    sign = "+" if parsed > 0 else ""
    return f"{sign}${parsed:,.2f}"


def pct(value: object, digits: int = 1) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "missing"
    if pd.isna(parsed):
        return "missing"
    return f"{parsed * 100:.{digits}f}%"


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def present(value: str | None) -> str:
    return "present" if bool(value) else "missing"


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    import yfinance as yf

    try:
        history = yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()
    return history if isinstance(history, pd.DataFrame) else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def cached_fundamental_data(symbol: str, fmp_key: str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bool]]:
    return fetch_fundamental_data(symbol, fmp_key)


@st.cache_data(ttl=900, show_spinner=False)
def cached_price_targets(symbol: str, fmp_key: str | None) -> dict[str, Any]:
    return fetch_price_target_consensus(fmp_key, symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_dcf_source_documents(symbol: str, fmp_key: str | None) -> pd.DataFrame:
    return fetch_dcf_source_documents(fmp_key, symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rapidapi_earnings_bundle(symbol: str, rapid_api_key: str | None) -> dict[str, Any]:
    return fetch_rapidapi_earnings_transcript_bundle(rapid_api_key, symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_peer_comparison_payload(symbol: str, fmp_key: str | None, cache_version: int = 3) -> dict[str, Any]:
    _ = cache_version
    peer_data = fetch_peer_comparison(fmp_key, symbol)
    return {
        "peer_symbols": list(peer_data.peer_symbols or []),
        "metrics": peer_data.metrics.copy() if isinstance(peer_data.metrics, pd.DataFrame) else pd.DataFrame(),
        "ratios": peer_data.ratios.copy() if isinstance(peer_data.ratios, pd.DataFrame) else pd.DataFrame(),
        "error": peer_data.error,
    }


def cached_peer_comparison(symbol: str, fmp_key: str | None) -> SimpleNamespace:
    payload = cached_peer_comparison_payload(symbol, fmp_key)
    return SimpleNamespace(
        peer_symbols=payload.get("peer_symbols", []),
        metrics=payload.get("metrics", pd.DataFrame()),
        ratios=payload.get("ratios", pd.DataFrame()),
        error=payload.get("error"),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def cached_directional_sentiment(symbol: str, fmp_key: str | None) -> dict[str, Any]:
    return fetch_directional_sentiment(fmp_key, symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news_articles(symbol: str, fmp_key: str | None) -> pd.DataFrame:
    return load_or_refresh_news_articles(symbol, fmp_key)


def news_articles_fingerprint(articles: pd.DataFrame) -> str:
    if articles is None or articles.empty:
        return "empty"
    columns = [column for column in ("published_at", "title", "url", "fetched_at") if column in articles.columns]
    if not columns:
        return f"rows={len(articles)}"
    hashed = pd.util.hash_pandas_object(articles[columns].astype(str), index=False)
    return f"rows={len(articles)};hash={int(hashed.sum())}"


@st.cache_data(ttl=900, show_spinner=False)
def cached_news_nlp(symbol: str, articles_fingerprint: str, _articles: pd.DataFrame) -> NewsNLPResult:
    _ = articles_fingerprint
    return analyze_news_articles(symbol, _articles)


@st.cache_data(ttl=300, show_spinner=False)
def cached_yahoo_expirations(symbol: str) -> list[str]:
    import yfinance as yf

    try:
        return list(yf.Ticker(symbol).options)
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def cached_massive_expirations(symbol: str, api_key: str | None) -> list[str]:
    if not api_key:
        return []
    try:
        return MassiveOptionsDataAdapter(api_key=api_key).get_option_expirations(symbol)
    except Exception:
        return []


def option_quotes_to_chain_frames(quotes: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for quote in quotes:
        contract = quote.contract
        bid = quote.quote.bid or 0.0
        ask = quote.quote.ask or 0.0
        last = quote.quote.last or quote.quote.mark or 0.0
        rows.append(
            {
                "contractSymbol": contract.symbol,
                "lastTradeDate": quote.quote.timestamp,
                "strike": contract.strike,
                "lastPrice": last,
                "bid": bid,
                "ask": ask,
                "change": 0.0,
                "percentChange": 0.0,
                "volume": quote.volume or 0.0,
                "openInterest": quote.open_interest or 0.0,
                "impliedVolatility": quote.implied_volatility or 0.0,
                "inTheMoney": False,
                "contractSize": "REGULAR",
                "currency": contract.currency,
                "option_type": contract.right.value,
                "delta": quote.delta,
                "gamma": quote.gamma,
                "theta": quote.theta,
                "vega": quote.vega,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    calls = frame[frame["option_type"] == "call"].copy()
    puts = frame[frame["option_type"] == "put"].copy()
    return calls.reset_index(drop=True), puts.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def cached_yahoo_option_chain(symbol: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    try:
        chain = yf.Ticker(symbol).option_chain(expiry)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return chain.calls, chain.puts


@st.cache_data(ttl=300, show_spinner=False)
def cached_massive_option_chain(
    symbol: str,
    expiry: str,
    api_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not api_key:
        return pd.DataFrame(), pd.DataFrame()
    try:
        request = OptionChainRequest(
            underlying=Instrument(symbol=symbol.upper(), asset_class=AssetClass.EQUITY),
            expiration=pd.to_datetime(expiry).date(),
        )
        quotes = MassiveOptionsDataAdapter(api_key=api_key).get_option_chain(request)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return option_quotes_to_chain_frames(quotes)


def massive_options_key(settings: Any) -> str | None:
    return settings.massive_api_key or settings.options_api_key or settings.polygon_api_key


def cached_expirations(symbol: str, settings: Any) -> tuple[list[str], str]:
    massive_expirations = cached_massive_expirations(symbol, massive_options_key(settings))
    if massive_expirations:
        return massive_expirations, "Massive"
    return cached_yahoo_expirations(symbol), "Yahoo"


def cached_option_chain(symbol: str, expiry: str, settings: Any) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    massive_calls, massive_puts = cached_massive_option_chain(symbol, expiry, massive_options_key(settings))
    if not massive_calls.empty or not massive_puts.empty:
        return massive_calls, massive_puts, "Massive"
    yahoo_calls, yahoo_puts = cached_yahoo_option_chain(symbol, expiry)
    return yahoo_calls, yahoo_puts, "Yahoo"


@st.cache_data(ttl=900, show_spinner=False)
def cached_options_history(symbol: str) -> pd.DataFrame:
    history = load_cached_market_history(symbol, lookback_days=800)
    if not history.empty:
        return history
    return cached_market_history(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_volatility_forecast_table(symbol: str, hold_days: int) -> pd.DataFrame:
    history = cached_options_history(symbol)
    horizons = tuple(sorted({*DEFAULT_VOL_FORECAST_HORIZONS, max(1, int(hold_days))}))
    forecasts = forecast_volatility_models(symbol, history, horizons=horizons)
    write_volatility_forecasts(forecasts_to_records(forecasts))
    return forecasts


def forecasts_to_records(frame: pd.DataFrame):
    from oqp.market import VolatilityForecast

    records = []
    for row in frame.to_dict("records"):
        records.append(
            VolatilityForecast(
                symbol=str(row.get("symbol") or ""),
                as_of=str(row.get("as_of") or ""),
                horizon_days=int(row.get("horizon_days") or 1),
                model=str(row.get("model") or ""),
                forecast_vol=safe_num(row.get("forecast_vol"), float("nan")),
                status=str(row.get("status") or ""),
                detail=str(row.get("detail") or ""),
                components=row.get("components") if isinstance(row.get("components"), dict) else {},
                created_at=str(row.get("created_at") or ""),
            )
        )
    return records


def latest_close(history: pd.DataFrame) -> float:
    if history.empty or "Close" not in history.columns:
        return 0.0
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    return float(close.iloc[-1]) if not close.empty else 0.0


def latest_rolling_value(series: pd.Series, window: int, *, fallback: float = 0.0, op: str = "mean") -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return fallback
    rolled = numeric.rolling(window).std().dropna() if op == "std" else numeric.rolling(window).mean().dropna()
    return float(rolled.iloc[-1]) if not rolled.empty else fallback


def estimate_atm_market_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, fallback: float) -> float:
    candidates: list[float] = []
    for chain in (calls, puts):
        if chain.empty or "strike" not in chain.columns or "impliedVolatility" not in chain.columns:
            continue
        local = chain.copy()
        local["distance"] = (pd.to_numeric(local["strike"], errors="coerce") - spot).abs()
        local["impliedVolatility"] = pd.to_numeric(local["impliedVolatility"], errors="coerce")
        local = local.dropna(subset=["distance", "impliedVolatility"])
        if not local.empty:
            candidates.append(float(local.sort_values("distance").iloc[0]["impliedVolatility"]))
    candidates = [value for value in candidates if value > 0]
    return float(sum(candidates) / len(candidates)) if candidates else fallback


def far_calendar_expiry(expirations: list[str], near_expiry: str, min_gap_days: int = 28) -> str | None:
    near_date = pd.to_datetime(near_expiry).date()
    for expiry in expirations:
        if (pd.to_datetime(expiry).date() - near_date).days >= min_gap_days:
            return expiry
    return None


def load_scratchpad() -> list[dict[str, Any]]:
    if not SCRATCHPAD_PATH.exists():
        return []
    try:
        payload = json.loads(SCRATCHPAD_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def save_scratchpad(rows: list[dict[str, Any]]) -> Path:
    SCRATCHPAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCRATCHPAD_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return SCRATCHPAD_PATH


def supply_chain_path(symbol: str) -> Path:
    safe_symbol = "".join(char for char in symbol.upper() if char.isalnum() or char in {"_", "-"})
    return SCRATCHPAD_PATH.parent / "supply_chain" / f"{safe_symbol}.json"


def default_supply_chain_rows(
    symbol: str,
    data: dict[str, Any],
    peers: list[str],
) -> list[dict[str, Any]]:
    company = str(data.get("company_name") or symbol.upper())
    sector = str(data.get("sector") or "Unknown")
    rows: list[dict[str, Any]] = [
        {
            "From": f"{sector} input suppliers",
            "To": company,
            "Relationship": "supplies inputs to",
            "Layer": "Upstream",
            "Weight": 2.0,
            "Evidence": "Starter placeholder. Replace with supplier/customer evidence from filings, contracts, or trusted data.",
            "Source": "Manual starter",
        },
        {
            "From": company,
            "To": "Distribution / sales channels",
            "Relationship": "sells through",
            "Layer": "Downstream",
            "Weight": 2.0,
            "Evidence": "Starter placeholder. Replace with disclosed channel relationships.",
            "Source": "Manual starter",
        },
        {
            "From": company,
            "To": "Customers / end markets",
            "Relationship": "serves",
            "Layer": "Downstream",
            "Weight": 3.0,
            "Evidence": "Starter placeholder. Replace with disclosed customer or segment evidence.",
            "Source": "Manual starter",
        },
    ]
    for peer in peers:
        rows.append(
            {
                "From": company,
                "To": peer,
                "Relationship": "competes with",
                "Layer": "Competitive Set",
                "Weight": 1.0,
                "Evidence": "Peer relationship returned by FMP stock-peers.",
                "Source": "FMP",
            }
        )
    return rows


def load_supply_chain_rows(symbol: str) -> list[dict[str, Any]]:
    path = supply_chain_path(symbol)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def save_supply_chain_rows(symbol: str, rows: list[dict[str, Any]]) -> Path:
    path = supply_chain_path(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


def render_supply_chain_graph(rows: pd.DataFrame) -> None:
    required = {"From", "To", "Weight"}
    if rows.empty or not required.issubset(rows.columns):
        st.info("Add at least one relationship row to render a supply-chain graph.")
        return

    graph = rows.copy()
    graph["From"] = graph["From"].astype(str).str.strip()
    graph["To"] = graph["To"].astype(str).str.strip()
    graph["Weight"] = pd.to_numeric(graph["Weight"], errors="coerce").fillna(1.0).clip(lower=0.1)
    graph = graph[(graph["From"] != "") & (graph["To"] != "")]
    if graph.empty:
        st.info("Add non-empty From and To values to render a supply-chain graph.")
        return

    labels = list(dict.fromkeys([*graph["From"].tolist(), *graph["To"].tolist()]))
    label_index = {label: index for index, label in enumerate(labels)}
    layer_colors = {
        "Upstream": "rgba(37, 99, 235, 0.35)",
        "Downstream": "rgba(22, 163, 74, 0.35)",
        "Competitive Set": "rgba(245, 158, 11, 0.35)",
        "Other": "rgba(100, 116, 139, 0.30)",
    }
    link_colors = [
        layer_colors.get(str(layer), layer_colors["Other"])
        for layer in graph.get("Layer", pd.Series(["Other"] * len(graph)))
    ]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(15, 23, 42, 0.25)", width=0.5),
                    label=labels,
                    color="rgba(37, 99, 235, 0.75)",
                ),
                link=dict(
                    source=[label_index[value] for value in graph["From"]],
                    target=[label_index[value] for value in graph["To"]],
                    value=graph["Weight"].tolist(),
                    color=link_colors,
                    customdata=graph[[col for col in ["Relationship", "Evidence", "Source"] if col in graph]].fillna(""),
                    hovertemplate="%{source.label} -> %{target.label}<br>Weight: %{value}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(height=460, margin=dict(t=20, b=20, l=10, r=10))
    style_dark_plotly(fig, hovermode=None)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def pct_change(close: pd.Series, periods: int) -> float:
    numeric = pd.to_numeric(close, errors="coerce").dropna()
    if len(numeric) <= periods:
        return float("nan")
    base = numeric.iloc[-periods - 1]
    if base == 0:
        return float("nan")
    return float((numeric.iloc[-1] / base) - 1)


def annualized_hv(close: pd.Series, window: int) -> float:
    numeric = pd.to_numeric(close, errors="coerce").dropna()
    if len(numeric) <= window:
        return float("nan")
    returns = np.log(numeric / numeric.shift(1)).dropna()
    hv = returns.tail(window).std() * np.sqrt(252)
    return float(hv) if pd.notna(hv) else float("nan")


def watchlist_opportunity_tag(row: dict[str, Any]) -> str:
    rsi = safe_num(row.get("RSI 14"), 50.0)
    ret_5d = safe_num(row.get("5D %"), 0.0)
    hv_21d = safe_num(row.get("HV 21D"), 0.0)
    volume_surge = safe_num(row.get("Vol / 20D"), 1.0)
    dist_high = safe_num(row.get("From 52W High"), 0.0)
    dist_ma20 = safe_num(row.get("Vs 20D MA"), 0.0)
    if rsi < 35 and dist_high < -0.10:
        return "oversold watch"
    if ret_5d > 0.06 and volume_surge > 1.5:
        return "momentum burst"
    if hv_21d > 0.60 and volume_surge > 1.3:
        return "high-vol setup"
    if dist_ma20 > 0.05 and ret_5d > 0:
        return "trend strength"
    if dist_high > -0.03:
        return "near high"
    return "neutral"


def format_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = safe_num(value, float("nan"))
    if pd.isna(number):
        return ""
    return f"{number:,.{digits}f}{suffix}"


def format_money_cell(value: Any, digits: int = 2) -> str:
    number = safe_num(value, float("nan"))
    if pd.isna(number):
        return ""
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.{digits}f}"


def format_percent_cell(value: Any, digits: int = 1) -> str:
    number = safe_num(value, float("nan"))
    if pd.isna(number):
        return ""
    return f"{number * 100:,.{digits}f}%"


def format_share_count(value: Any) -> str:
    number = safe_num(value, float("nan"))
    if pd.isna(number):
        return ""
    if abs(number) >= 1e9:
        return f"{number / 1e9:,.2f}B"
    if abs(number) >= 1e6:
        return f"{number / 1e6:,.2f}M"
    return f"{number:,.0f}"


def format_dcf_bridge_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()

    def _format_amount(row: pd.Series) -> str:
        metric = str(row.get("Metric") or "")
        amount = row.get("Amount")
        if "Shares" in metric:
            return format_share_count(amount)
        if "Share" in metric:
            return money(amount)
        return format_compact_currency(safe_num(amount))

    display["Amount"] = display.apply(_format_amount, axis=1)
    return display


def format_dcf_source_documents_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Source", "Document", "Date", "Title", "URL"])
    display = frame.copy()
    keep = ["Source", "Document", "Date", "Title", "URL", "Text Preview"]
    display = display[[column for column in keep if column in display.columns]]
    if "Text Preview" in display:
        display["Text Preview"] = display["Text Preview"].astype(str).str.slice(0, 220)
    return display


def format_management_tone_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Lens", "Tone", "Score", "Evidence", "Bullish Terms", "Bearish Terms"])
    display = frame.copy()
    if "Score" in display:
        display["Score"] = display["Score"].map(lambda value: format_float(value, 2))
    return display


def normalize_evidence_list(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value if str(item).strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def format_ai_horizon_frame(analysis: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, label in (
        ("short_term_outlook", "Short Term"),
        ("mid_term_outlook", "Mid Term"),
        ("long_term_outlook", "Long Term"),
    ):
        item = analysis.get(key) if isinstance(analysis.get(key), dict) else {}
        rows.append(
            {
                "Horizon": item.get("horizon") or label,
                "Label": item.get("label") or "missing",
                "Confidence": format_float(item.get("confidence"), 2),
                "Summary": item.get("summary") or "",
                "Evidence": normalize_evidence_list(item.get("evidence")),
            }
        )
    return pd.DataFrame(rows)


def format_ai_key_value_frame(value: Any, *, key_label: str = "Input", value_label: str = "Guidance") -> pd.DataFrame:
    if not isinstance(value, dict) or not value:
        return pd.DataFrame(columns=[key_label, value_label])
    return pd.DataFrame(
        [
            {key_label: str(key).replace("_", " ").title(), value_label: normalize_evidence_list(item)}
            for key, item in value.items()
        ]
    )


def format_ai_list_frame(items: Any, *, column: str) -> pd.DataFrame:
    if not isinstance(items, list):
        items = [items] if items else []
    return pd.DataFrame([{column: str(item)} for item in items if str(item).strip()])


def format_watchlist_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format watchlist metrics for scanning instead of raw precision."""

    display = frame.copy()
    percent_columns: dict[str, int] = {
        "1D %": 2,
        "5D %": 2,
        "20D %": 2,
        "HV 21D": 1,
        "HV 63D": 1,
        "Vs 20D MA": 2,
        "From 52W High": 2,
        "ATR %": 2,
    }
    number_formats: dict[str, tuple[int, str]] = {
        "Last": (2, ""),
        "RSI 14": (1, ""),
        "Vol / 20D": (2, "x"),
    }
    for column, digits in percent_columns.items():
        if column in display:
            display[column] = display[column].map(lambda value, d=digits: format_percent_cell(value, d))
    for column, (digits, suffix) in number_formats.items():
        if column in display:
            display[column] = display[column].map(lambda value, d=digits, s=suffix: format_float(value, d, s))
    if "As Of" in display:
        display["As Of"] = pd.to_datetime(display["As Of"], errors="coerce").dt.date.astype("string")
        display["As Of"] = display["As Of"].replace("<NA>", "")
    return display


def watchlist_symbol_link(symbol: str) -> str:
    symbol_key = str(symbol or "").upper().strip()
    query = f"workbench_view=Opportunity%20Hub&hub_symbol={quote(symbol_key)}"
    try:
        current_url = str(st.context.url or "")
    except Exception:
        current_url = ""
    if current_url.startswith("http"):
        return f"{current_url.split('?', 1)[0]}?{query}"
    return f"?{query}"


def workbench_query_link(**params: str) -> str:
    query = "&".join(
        f"{quote(str(key))}={quote(str(value))}"
        for key, value in params.items()
        if value is not None and str(value) != ""
    )
    return f"?{query}" if query else "?"


def render_workbench_nav(active_view: str) -> None:
    links = []
    for view in WORKBENCH_VIEWS:
        active_class = " is-active" if view == active_view else ""
        href = workbench_query_link(workbench_view=view)
        safe_href = escape(href, quote=True)
        links.append(
            f'<a class="{active_class}" href="{safe_href}" target="_self" '
            f'onclick="window.location.href=this.href; return false;">{escape(workbench_view_label(view))}</a>'
        )
    st.markdown(f'<nav class="oqp-workbench-nav">{"".join(links)}</nav>', unsafe_allow_html=True)


def sorted_watchlist_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if frame.empty:
        return frame.copy(), "", "asc"
    sort_col = str(st.query_params.get("watchlist_sort_col") or "").strip()
    sort_dir = str(st.query_params.get("watchlist_sort_dir") or "asc").lower().strip()
    if sort_col not in frame.columns:
        return frame.copy(), "", "asc"
    ascending = sort_dir != "desc"
    sorted_frame = frame.copy()
    if sort_col == "As Of":
        sorted_frame["_sort_value"] = pd.to_datetime(sorted_frame[sort_col], errors="coerce")
    elif sort_col in {"Symbol", "Opportunity"}:
        sorted_frame["_sort_value"] = sorted_frame[sort_col].astype(str)
    else:
        sorted_frame["_sort_value"] = pd.to_numeric(sorted_frame[sort_col], errors="coerce")
    sorted_frame = (
        sorted_frame.sort_values("_sort_value", ascending=ascending, na_position="last")
        .drop(columns=["_sort_value"])
        .reset_index(drop=True)
    )
    return sorted_frame, sort_col, "asc" if ascending else "desc"


def watchlist_aggrid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare watchlist data for an interactive browser-side grid."""

    if frame.empty:
        return pd.DataFrame()
    display = frame.copy()
    display.insert(0, "Keep", True)
    display["Symbol"] = display["Symbol"].astype(str).str.upper().str.strip()
    percent_columns = [
        "1D %",
        "5D %",
        "20D %",
        "HV 21D",
        "HV 63D",
        "Vs 20D MA",
        "From 52W High",
        "ATR %",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce") * 100
    numeric_columns = ["Last", "RSI 14", "Vol / 20D"]
    for column in numeric_columns:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce")
    if "As Of" in display:
        display["As Of"] = pd.to_datetime(display["As Of"], errors="coerce").dt.strftime("%Y-%m-%d")
        display["As Of"] = display["As Of"].fillna("")
    columns = [
        "Keep",
        "Symbol",
        "Last",
        "1D %",
        "5D %",
        "20D %",
        "HV 21D",
        "HV 63D",
        "RSI 14",
        "Vs 20D MA",
        "From 52W High",
        "ATR %",
        "Vol / 20D",
        "As Of",
        "Opportunity",
    ]
    return display[[column for column in columns if column in display.columns]]


def aggrid_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def render_watchlist_aggrid(frame: pd.DataFrame) -> bool:
    """Render the watchlist using streamlit-aggrid when available."""

    if AgGrid is None or GridOptionsBuilder is None or GridUpdateMode is None or DataReturnMode is None or JsCode is None:
        return False
    grid_frame = watchlist_aggrid_frame(frame)
    if grid_frame.empty:
        st.info("No cached market rows yet. Refresh stale/missing symbols to populate the ranking table.")
        return True

    money_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "";
            return "$" + Number(params.value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        """
    )
    pct_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "";
            return Number(params.value).toFixed(2) + "%";
        }
        """
    )
    pct_1_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "";
            return Number(params.value).toFixed(1) + "%";
        }
        """
    )
    num_1_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "";
            return Number(params.value).toFixed(1);
        }
        """
    )
    ratio_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "";
            return Number(params.value).toFixed(2) + "x";
        }
        """
    )
    symbol_style = JsCode(
        """
        function(params) {
            return {
                color: "#60a5fa",
                fontWeight: "800",
                cursor: "pointer"
            };
        }
        """
    )
    opportunity_style = JsCode(
        """
        function(params) {
            return {
                color: "#dbeafe",
                fontWeight: "700"
            };
        }
        """
    )
    heart_renderer = JsCode(
        """
        class HeartRenderer {
            init(params) {
                this.params = params;
                this.button = document.createElement("button");
                this.button.type = "button";
                this.button.style.border = "0";
                this.button.style.background = "transparent";
                this.button.style.cursor = "pointer";
                this.button.style.fontSize = "18px";
                this.button.style.lineHeight = "1";
                this.button.style.padding = "0";
                this.button.style.width = "100%";
                this.button.style.height = "100%";
                this.button.title = "Click to remove from watchlist";
                this.button.addEventListener("click", (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    params.node.setDataValue(params.column.colId, !params.value);
                });
                this.refresh(params);
            }
            getGui() {
                return this.button;
            }
            refresh(params) {
                this.params = params;
                const kept = params.value !== false;
                this.button.textContent = kept ? "♥" : "♡";
                this.button.style.color = kept ? "#fb7185" : "#64748b";
                return true;
            }
        }
        """
    )

    builder = GridOptionsBuilder.from_dataframe(grid_frame)
    builder.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        menuTabs=["filterMenuTab", "generalMenuTab"],
    )
    builder.configure_selection(
        selection_mode="single",
        use_checkbox=False,
        suppressRowClickSelection=False,
    )
    builder.configure_grid_options(
        rowHeight=42,
        headerHeight=42,
        suppressCellFocus=True,
        enableCellTextSelection=True,
        animateRows=True,
        suppressRowHoverHighlight=False,
        tooltipShowDelay=250,
    )
    builder.configure_column(
        "Keep",
        header_name="♥",
        editable=True,
        pinned="left",
        width=76,
        filter=False,
        sortable=False,
        cellRenderer=heart_renderer,
        cellStyle={"display": "flex", "alignItems": "center", "justifyContent": "center"},
    )
    builder.configure_column(
        "Symbol",
        pinned="left",
        width=95,
        cellStyle=symbol_style,
        tooltipField="Symbol",
    )
    if "Last" in grid_frame:
        builder.configure_column("Last", type=["numericColumn"], valueFormatter=money_formatter, width=105)
    for column in ["1D %", "5D %", "20D %", "Vs 20D MA", "ATR %"]:
        if column in grid_frame:
            builder.configure_column(column, type=["numericColumn"], valueFormatter=pct_formatter, width=112)
    if "From 52W High" in grid_frame:
        builder.configure_column(
            "From 52W High",
            type=["numericColumn"],
            valueFormatter=pct_formatter,
            width=155,
            minWidth=145,
            headerTooltip="Distance from 52-week high",
        )
    for column in ["HV 21D", "HV 63D"]:
        if column in grid_frame:
            builder.configure_column(column, type=["numericColumn"], valueFormatter=pct_1_formatter, width=105)
    if "RSI 14" in grid_frame:
        builder.configure_column("RSI 14", type=["numericColumn"], valueFormatter=num_1_formatter, width=92)
    if "Vol / 20D" in grid_frame:
        builder.configure_column("Vol / 20D", type=["numericColumn"], valueFormatter=ratio_formatter, width=105)
    if "As Of" in grid_frame:
        builder.configure_column("As Of", width=112)
    if "Opportunity" in grid_frame:
        builder.configure_column("Opportunity", width=165, cellStyle=opportunity_style)

    custom_css = {
        ".ag-root-wrapper": {
            "background-color": "#07101a !important",
            "border": "1px solid rgba(72, 92, 122, 0.30) !important",
            "border-radius": "10px !important",
            "box-shadow": "0 24px 55px rgba(0, 0, 0, 0.20) !important",
        },
        ".ag-header": {
            "background-color": "#0f1826 !important",
            "border-bottom": "1px solid rgba(93, 113, 143, 0.30) !important",
        },
        ".ag-header-cell-label": {
            "color": "#8fa2bd !important",
            "font-weight": "800 !important",
            "letter-spacing": "0.04em !important",
        },
        ".ag-row": {
            "background-color": "#08101a !important",
            "border-bottom": "1px solid rgba(65, 84, 112, 0.16) !important",
        },
        ".ag-row-odd": {
            "background-color": "#0c1521 !important",
        },
        ".ag-row-hover": {
            "background-color": "#152336 !important",
        },
        ".ag-cell": {
            "color": "#dbeafe !important",
            "border-color": "rgba(65, 84, 112, 0.16) !important",
            "font-size": "0.84rem !important",
        },
        ".ag-checkbox-input-wrapper.ag-checked::after": {
            "color": "#fb7185 !important",
        },
        ".ag-paging-panel": {
            "background-color": "#07101a !important",
            "color": "#9fb0c7 !important",
            "border-top": "1px solid rgba(65, 84, 112, 0.20) !important",
        },
    }

    response = AgGrid(
        grid_frame,
        gridOptions=builder.build(),
        height=min(620, max(360, 112 + len(grid_frame) * 42)),
        theme="dark",
        update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
        update_on=["cellValueChanged", "selectionChanged"],
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        allow_unsafe_jscode=True,
        custom_css=custom_css,
        show_search=True,
        show_download_button=False,
        key=f"watchlist_aggrid_{hash(tuple(grid_frame['Symbol'].tolist()))}",
    )

    returned_rows = aggrid_rows(getattr(response, "data", None))
    removed = [
        str(row.get("Symbol") or "").upper().strip()
        for row in returned_rows
        if not bool(row.get("Keep", True)) and str(row.get("Symbol") or "").strip()
    ]
    if removed:
        for removed_symbol in removed:
            remove_stock_watchlist_symbol(removed_symbol)
        cached_watchlist_snapshot.clear()
        st.toast(f"Removed {', '.join(removed)} from watchlist.")
        st.rerun()

    selected_rows = aggrid_rows(getattr(response, "selected_rows", None))
    if selected_rows:
        selected_symbol = str(selected_rows[0].get("Symbol") or "").upper().strip()
        if selected_symbol:
            st.session_state["hub_symbol"] = selected_symbol
            st.session_state["workbench_symbol"] = selected_symbol
            st.session_state["workbench_view"] = "Opportunity Hub"
            st.rerun()

    st.caption("Sort and filter inside the grid. Edit the heart checkbox to remove. Click a row to open the Opportunity Hub.")
    return True


def render_watchlist_dark_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No cached market rows yet. Refresh stale/missing symbols to populate the ranking table.")
        return

    sorted_frame, sort_col, sort_dir = sorted_watchlist_frame(frame)
    display = format_watchlist_display(sorted_frame)
    columns = [
        "Symbol",
        "Last",
        "1D %",
        "5D %",
        "20D %",
        "HV 21D",
        "HV 63D",
        "RSI 14",
        "Vs 20D MA",
        "From 52W High",
        "ATR %",
        "Vol / 20D",
        "As Of",
        "Opportunity",
    ]
    columns = [column for column in columns if column in display.columns]

    header_cells = ['<th title="Click the heart to remove from watchlist">♥</th>']
    for column in columns:
        next_dir = "desc" if sort_col == column and sort_dir == "asc" else "asc"
        arrow = " ↓" if sort_col == column and sort_dir == "desc" else " ↑" if sort_col == column else ""
        css_class = "active-sort" if sort_col == column else ""
        href = workbench_query_link(
            workbench_view="Watchlist",
            watchlist_sort_col=column,
            watchlist_sort_dir=next_dir,
        )
        align_class = "symbol-header" if column == "Symbol" else ""
        safe_href = escape(href, quote=True)
        header_cells.append(
            f'<th class="{align_class}"><a class="{css_class}" href="{safe_href}" target="_self" '
            f'onclick="window.location.href=this.href; return false;">{escape(column)}{arrow}</a></th>'
        )

    row_html: list[str] = []
    for _, row in display.iterrows():
        symbol = str(row.get("Symbol") or "").upper().strip()
        remove_href = workbench_query_link(workbench_view="Watchlist", remove_watchlist_symbol=symbol)
        safe_remove_href = escape(remove_href, quote=True)
        cells = [
            (
                f'<td><a class="heart-link" href="{safe_remove_href}" target="_self" '
                f'onclick="window.location.href=this.href; return false;" '
                f'title="Remove {escape(symbol)} from watchlist">♥</a></td>'
            )
        ]
        for column in columns:
            value = "" if pd.isna(row.get(column)) else str(row.get(column))
            if column == "Symbol":
                safe_symbol_href = escape(watchlist_symbol_link(symbol), quote=True)
                cells.append(
                    f'<td><a class="symbol-link" href="{safe_symbol_href}" target="_self" '
                    f'onclick="window.location.href=this.href; return false;">{escape(value)}</a></td>'
                )
            elif column == "Opportunity":
                cells.append(f'<td class="opportunity-cell">{escape(value)}</td>')
            else:
                cells.append(f"<td>{escape(value)}</td>")
        row_html.append(f"<tr>{''.join(cells)}</tr>")

    html = (
        '<div class="oqp-watchlist-table-wrap">'
        '<table class="oqp-watchlist-table">'
        f"<thead><tr>{''.join(header_cells)}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Headers sort the local cache. Tickers open the Opportunity Hub. Hearts remove symbols.")


def watchlist_editor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a sortable/editable watchlist table while preserving raw numeric values."""

    if frame.empty:
        return pd.DataFrame()
    display = frame.copy()
    display["Ticker"] = display["Symbol"].astype(str).str.upper().str.strip()
    display["Symbol"] = display["Ticker"].map(watchlist_symbol_link)
    display.insert(0, "♥", True)
    percent_columns = [
        "1D %",
        "5D %",
        "20D %",
        "HV 21D",
        "HV 63D",
        "Vs 20D MA",
        "From 52W High",
        "ATR %",
    ]
    for column in percent_columns:
        if column in display:
            display[column] = pd.to_numeric(display[column], errors="coerce") * 100
    if "As Of" in display:
        display["As Of"] = pd.to_datetime(display["As Of"], errors="coerce").dt.date
    columns = [
        "♥",
        "Symbol",
        "Ticker",
        "Last",
        "1D %",
        "5D %",
        "20D %",
        "HV 21D",
        "HV 63D",
        "RSI 14",
        "Vs 20D MA",
        "From 52W High",
        "ATR %",
        "Vol / 20D",
        "As Of",
        "Opportunity",
    ]
    return display[[column for column in columns if column in display.columns]]


def watchlist_column_config() -> dict[str, Any]:
    return {
        "♥": st.column_config.CheckboxColumn(
            "♥",
            help="Uncheck to remove this ticker from the watchlist.",
            width="small",
            default=True,
            pinned=True,
        ),
        "Symbol": st.column_config.LinkColumn(
            "Symbol",
            help="Click the ticker to open it in the Opportunity Hub.",
            display_text=r".*hub_symbol=([^&]+).*",
            pinned=True,
            width="small",
        ),
        "Ticker": None,
        "Last": st.column_config.NumberColumn("Last", format="$%.2f"),
        "1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
        "5D %": st.column_config.NumberColumn("5D %", format="%.2f%%"),
        "20D %": st.column_config.NumberColumn("20D %", format="%.2f%%"),
        "HV 21D": st.column_config.NumberColumn("HV 21D", format="%.1f%%"),
        "HV 63D": st.column_config.NumberColumn("HV 63D", format="%.1f%%"),
        "RSI 14": st.column_config.NumberColumn("RSI 14", format="%.1f"),
        "Vs 20D MA": st.column_config.NumberColumn("Vs 20D MA", format="%.2f%%"),
        "From 52W High": st.column_config.NumberColumn("From 52W High", format="%.2f%%"),
        "ATR %": st.column_config.NumberColumn("ATR %", format="%.2f%%"),
        "Vol / 20D": st.column_config.NumberColumn("Vol / 20D", format="%.2fx"),
        "As Of": st.column_config.DateColumn("As Of"),
        "Opportunity": st.column_config.TextColumn("Opportunity"),
    }


def format_options_scanner_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format option scanner metrics for quick visual triage."""

    display = frame.copy()
    percent_columns = ("PoP", "IV Edge", "Market IV")
    money_columns = ("EV", "VaR 95", "Max Profit", "Max Loss", "Debit/Credit", "Edge", "Mid")
    number_columns = ("Strike", "Width")

    for column in percent_columns:
        if column in display:
            display[column] = display[column].map(lambda value: format_percent_cell(value, 1))
    for column in money_columns:
        if column in display:
            display[column] = display[column].map(format_money_cell)
    for column in number_columns:
        if column in display:
            display[column] = display[column].map(lambda value: format_float(value, 2))
    if "EV / Max Loss" in display:
        display["EV / Max Loss"] = display["EV / Max Loss"].map(lambda value: format_float(value, 2, "x"))
    return display


def format_options_playbook_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Theme", "Fit", "Best Candidate", "PoP", "EV", "Max Loss", "Why", "Reference", "Source"])
    display = frame.copy()
    if "PoP" in display:
        display["PoP"] = display["PoP"].map(lambda value: format_percent_cell(value, 1))
    for column in ("EV", "Max Loss"):
        if column in display:
            display[column] = display[column].map(format_money_cell)
    columns = ["Theme", "Fit", "Best Candidate", "PoP", "EV", "Max Loss", "Why", "Reference", "Source"]
    return display[[column for column in columns if column in display.columns]]


def format_vol_forecast_display(frame: pd.DataFrame, *, hold_days: int) -> pd.DataFrame:
    """Format volatility model rows for the Options Scanner."""

    if frame.empty:
        return pd.DataFrame(columns=["Horizon", "Model", "Forecast Vol", "Status", "Detail"])
    display = frame.copy()
    display = display[display["horizon_days"].isin([1, 5, 21, int(hold_days)])].copy()
    display["Horizon"] = display["horizon_days"].map(lambda value: f"{int(value)}D")
    display["Model"] = display["model"].map(
        {
            "baseline_blend": "Baseline",
            "har_hv": "HAR-HV",
            "garch_1_1": "GARCH(1,1)",
            "ensemble": "Ensemble",
        }
    ).fillna(display["model"])
    display["Forecast Vol"] = display["forecast_vol"].map(lambda value: format_float(safe_num(value, float("nan")) * 100, 1, "%"))
    display["Status"] = display["status"].astype(str)
    display["Detail"] = display["detail"].astype(str)
    return display[["Horizon", "Model", "Forecast Vol", "Status", "Detail"]]


def format_directional_horizon_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format directional model rows for dashboard display."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Horizon",
                "Direction",
                "Score",
                "Confidence",
                "Expected Move",
                "Trend",
                "RSI",
                "Sentiment",
                "Analyst",
                "Options",
            ]
        )
    display = frame.copy()
    for column in ("Score", "Trend Score", "RSI Score", "Sentiment Score", "Analyst Score", "Options Score"):
        if column in display:
            display[column] = display[column].map(lambda value: format_float(value, 2))
    for column in ("Confidence", "Expected Move", "Data Coverage"):
        if column in display:
            display[column] = display[column].map(lambda value: format_percent_cell(value, 1))
    rename = {
        "Trend Score": "Trend",
        "RSI Score": "RSI",
        "Sentiment Score": "Sentiment",
        "Analyst Score": "Analyst",
        "Options Score": "Options",
    }
    display = display.rename(columns=rename)
    columns = [
        "Horizon",
        "Direction",
        "Score",
        "Confidence",
        "Expected Move",
        "Trend",
        "RSI",
        "Sentiment",
        "Analyst",
        "Options",
        "Detail",
    ]
    return display[[column for column in columns if column in display.columns]]


def format_directional_contribution_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Horizon", "Signal", "Score", "Weight", "Contribution", "Detail"])
    display = frame.copy()
    for column in ("Score", "Contribution"):
        if column in display:
            display[column] = display[column].map(lambda value: format_float(value, 2))
    if "Weight" in display:
        display["Weight"] = display["Weight"].map(lambda value: format_percent_cell(value, 0))
    columns = ["Horizon", "Signal", "Score", "Weight", "Contribution", "Detail"]
    return display[[column for column in columns if column in display.columns]]


def format_sentiment_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Source", "Date", "Title", "Sentiment", "Score"])
    display = frame.copy()
    if "Score" in display:
        display["Score"] = display["Score"].map(lambda value: format_float(value, 2))
    columns = ["Source", "Date", "Title", "Sentiment", "Score"]
    return display[[column for column in columns if column in display.columns]].head(30)


def format_news_article_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Published", "Title", "Site", "Sentiment", "Score", "Topics", "URL"])
    display = frame.copy()
    if "Score" in display:
        display["Score"] = display["Score"].map(lambda value: format_float(value, 2))
    columns = ["Published", "Title", "Site", "Sentiment", "Score", "Topics", "Bullish Terms", "Bearish Terms", "URL"]
    return display[[column for column in columns if column in display.columns]]


def format_news_topic_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Topic", "Count", "Avg Sentiment"])
    display = frame.copy()
    if "Avg Sentiment" in display:
        display["Avg Sentiment"] = display["Avg Sentiment"].map(lambda value: format_float(value, 2))
    return display


def format_news_catalyst_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Catalyst", "Articles", "Tone", "Avg Score", "Latest", "Latest Headline", "Evidence", "Action Read"])
    display = frame.copy()
    if "Avg Score" in display:
        display["Avg Score"] = display["Avg Score"].map(lambda value: format_float(value, 2))
    columns = ["Catalyst", "Articles", "Tone", "Avg Score", "Latest", "Latest Headline", "Evidence", "Action Read"]
    return display[[column for column in columns if column in display.columns]]


def format_news_evidence_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Published", "Catalyst", "Tone", "Score", "Headline", "Evidence"])
    display = frame.copy()
    if "Score" in display:
        display["Score"] = display["Score"].map(lambda value: format_float(value, 2))
    columns = ["Published", "Catalyst", "Tone", "Score", "Headline", "Evidence"]
    return display[[column for column in columns if column in display.columns]]


def format_provider_status_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Provider", "Role", "Status", "Detail"])
    columns = ["Provider", "Role", "Status", "Detail"]
    return frame[[column for column in columns if column in frame.columns]].copy()


def format_opportunity_lens_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Lens", "Score", "Bias", "Detail"])
    display = frame.copy()
    if "Score" in display:
        display["Score"] = display["Score"].map(lambda value: format_float(value, 2))
    columns = ["Lens", "Score", "Bias", "Detail"]
    return display[[column for column in columns if column in display.columns]]


def format_vehicle_route_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Vehicle", "Fit", "Current Read", "Risk Check"])
    columns = ["Vehicle", "Fit", "Current Read", "Risk Check"]
    return frame[[column for column in columns if column in frame.columns]].copy()


def format_decision_checklist_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Check", "Status", "Detail"])
    columns = ["Check", "Status", "Detail"]
    return frame[[column for column in columns if column in frame.columns]].copy()


def format_saved_ideas_display(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["created_at", "symbol", "status", "vehicle", "conviction", "primary_route"])
    frame = pd.DataFrame(rows)
    columns = [
        "created_at",
        "symbol",
        "status",
        "vehicle",
        "conviction",
        "primary_route",
        "direction",
        "horizon",
        "trigger",
        "invalidation",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    if "created_at" in display:
        display["created_at"] = display["created_at"].astype(str).str.replace("T", " ", regex=False)
    return display


def format_opportunity_history_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "captured_at",
                "symbol",
                "spot",
                "action_bucket",
                "primary_route",
                "direction",
                "target_upside",
                "forecast_vol",
                "market_iv",
                "news_tone",
            ]
        )
    display = frame.copy()
    if "captured_at" in display:
        display["captured_at"] = display["captured_at"].astype(str).str.replace("T", " ", regex=False)
    if "spot" in display:
        display["spot"] = display["spot"].map(format_money_cell)
    for column in ("target_upside", "forecast_vol", "market_iv"):
        if column in display:
            display[column] = display[column].map(lambda value: format_percent_cell(value, 1))
    if "direction_score" in display:
        display["direction_score"] = display["direction_score"].map(lambda value: format_float(value, 2))
    if "news_score" in display:
        display["news_score"] = display["news_score"].map(lambda value: format_float(value, 2))
    columns = [
        "captured_at",
        "symbol",
        "spot",
        "action_bucket",
        "primary_route",
        "direction",
        "direction_score",
        "target_upside",
        "forecast_vol",
        "market_iv",
        "news_tone",
        "news_score",
        "reference_expiry",
    ]
    return display[[column for column in columns if column in display.columns]]


def format_cache_status_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Format market-cache status rows for compact display."""

    display = frame.copy()
    if "Age Hours" in display:
        display["Age Hours"] = display["Age Hours"].map(lambda value: format_float(value, 1, "h"))
    return display


@st.cache_data(ttl=900, show_spinner=False)
def cached_watchlist_snapshot(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        history = load_cached_market_history(symbol, lookback_days=370)
        if history.empty or "Close" not in history.columns:
            rows.append({"Symbol": symbol, "Opportunity": "no data"})
            continue
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if close.empty:
            rows.append({"Symbol": symbol, "Opportunity": "no data"})
            continue
        last_price = float(close.iloc[-1])
        snapshot = volatility_snapshot(history)
        ma20 = close.tail(20).mean() if len(close) >= 20 else float("nan")
        high_52w = close.max()
        volume = pd.to_numeric(history.get("Volume", pd.Series(dtype=float)), errors="coerce").dropna()
        volume_20d = volume.tail(20).mean() if len(volume) >= 20 else float("nan")
        last_volume = volume.iloc[-1] if not volume.empty else float("nan")
        row: dict[str, Any] = {
            "Symbol": symbol,
            "Last": last_price,
            "1D %": pct_change(close, 1),
            "5D %": pct_change(close, 5),
            "20D %": pct_change(close, 20),
            "HV 21D": annualized_hv(close, 21),
            "HV 63D": annualized_hv(close, 63),
            "RSI 14": snapshot.rsi_14,
            "Vs 20D MA": (last_price / ma20) - 1 if ma20 and pd.notna(ma20) else float("nan"),
            "From 52W High": (last_price / high_52w) - 1 if high_52w and pd.notna(high_52w) else float("nan"),
            "ATR %": (snapshot.atr_14 / last_price) if last_price > 0 else float("nan"),
            "Vol / 20D": (last_volume / volume_20d) if volume_20d and pd.notna(volume_20d) else float("nan"),
            "As Of": history.index[-1],
        }
        row["Opportunity"] = watchlist_opportunity_tag(row)
        rows.append(row)
    return pd.DataFrame(rows)


def render_watchlist_tab() -> None:
    remove_symbol = str(st.query_params.get("remove_watchlist_symbol") or "").upper().strip()
    if remove_symbol:
        remove_stock_watchlist_symbol(remove_symbol)
        cached_watchlist_snapshot.clear()
        try:
            st.query_params.clear()
        except Exception:
            pass
        st.toast(f"Removed {remove_symbol} from watchlist.")
        st.rerun()

    watchlist = load_stock_watchlist()
    st.subheader("Watchlist")
    cols = st.columns([1.2, 1, 2.8])
    symbol = cols[0].text_input("Symbol", value=st.session_state.get("workbench_symbol", "AAPL")).upper().strip()
    if cols[1].button("Add To Watchlist", use_container_width=True, disabled=not symbol):
        add_stock_watchlist_symbol(symbol)
        st.session_state["workbench_symbol"] = symbol
        st.rerun()
    cols[2].caption("Click a ticker to open the Opportunity Hub. Click the heart to remove it.")

    st.caption(f"Watchlist path: `{display_path(DEFAULT_STOCK_WATCHLIST_PATH)}`")
    if not watchlist:
        st.info("No watchlist symbols yet.")
        return

    cache_status = market_cache_status(watchlist)
    stale_symbols = cache_status.loc[
        cache_status["State"].isin(["missing", "stale"]),
        "Symbol",
    ].astype(str).tolist()
    state_counts = cache_status["State"].value_counts().to_dict()
    st.caption(
        "Market cache: "
        f"`{display_path(DEFAULT_MARKET_CACHE_PATH)}` | "
        f"fresh={state_counts.get('fresh', 0)} "
        f"stale={state_counts.get('stale', 0)} "
        f"missing={state_counts.get('missing', 0)} | "
        f"stale threshold={DEFAULT_MARKET_CACHE_MAX_AGE_HOURS:.0f}h"
    )

    refresh_cols = st.columns([1.1, 1.1, 2.8])
    refresh_stale = refresh_cols[0].button(
        "Refresh Stale/Missing",
        use_container_width=True,
        disabled=not stale_symbols,
    )
    refresh_all = refresh_cols[1].button("Refresh Full Watchlist", use_container_width=True)
    refresh_cols[2].caption(
        "The table reads local SQLite first. Refresh buttons are the only network calls on this tab."
    )

    symbols_to_refresh = stale_symbols if refresh_stale else (watchlist if refresh_all else [])
    if symbols_to_refresh:
        estimated = max(8, min(90, len(symbols_to_refresh) * 4))
        progress, status = progress_bar(
            f"Refreshing {len(symbols_to_refresh)} watchlist symbol(s)",
            estimate_seconds=estimated,
        )
        progress_step(progress, status, 12, "Starting Yahoo market-cache refresh...")
        refresh_result = refresh_yahoo_market_cache(tuple(symbols_to_refresh), period="1y")
        progress_step(progress, status, 88, "Writing refreshed market data to SQLite cache...")
        cached_watchlist_snapshot.clear()
        cache_status = market_cache_status(watchlist)
        ok_count = int(refresh_result["Status"].eq("ok").sum()) if "Status" in refresh_result else 0
        progress_step(progress, status, 100, f"Refresh complete: {ok_count}/{len(symbols_to_refresh)} updated.")
        finish_progress(progress, status)
        st.success(f"Market cache refresh complete: {ok_count}/{len(symbols_to_refresh)} updated.")
        render_dark_table(refresh_result, max_height_px=260)

    frame = cached_watchlist_snapshot(tuple(watchlist))
    if not render_watchlist_aggrid(frame):
        st.warning("`streamlit-aggrid` is not installed, so the watchlist is using the static dark-table fallback.")
        render_watchlist_dark_table(frame)
    with st.expander("Market cache status", expanded=False):
        render_dark_table(format_cache_status_display(cache_status), max_height_px=320)


def render_stock_snapshot(symbol: str, data: dict[str, Any]) -> None:
    metrics = st.columns(6)
    metrics[0].metric("Price", money(data.get("price")))
    metrics[1].metric("Market Cap", format_compact_currency(safe_num(data.get("market_cap"))))
    metrics[2].metric("P/E", f"{safe_num(data.get('pe')):.2f}")
    metrics[3].metric("PEG", f"{safe_num(data.get('peg')):.2f}")
    metrics[4].metric("Sector", str(data.get("sector") or "Unknown"))
    metrics[5].metric("FCF TTM", format_compact_currency(safe_num(data.get("fcf_ttm"))))

    hist = data.get("hist_df")
    if isinstance(hist, pd.DataFrame) and not hist.empty and "Close" in hist:
        chart = hist.copy()
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            row_heights=[0.52, 0.18, 0.30],
            subplot_titles=("Price", "Volume", "MACD"),
        )
        if "BB_Upper" in chart and "BB_Lower" in chart:
            fig.add_scatter(
                x=chart.index,
                y=chart["BB_Upper"],
                name="Bollinger Upper",
                line=dict(width=1, color="#93C5FD"),
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=chart.index,
                y=chart["BB_Lower"],
                name="Bollinger Lower",
                fill="tonexty",
                fillcolor="rgba(147, 197, 253, 0.18)",
                line=dict(width=1, color="#93C5FD"),
                row=1,
                col=1,
            )
        if "SMA_20" in chart:
            fig.add_scatter(
                x=chart.index,
                y=chart["SMA_20"],
                name="Bollinger Mid",
                line=dict(width=1.2, color="#60A5FA", dash="dot"),
                row=1,
                col=1,
            )
        fig.add_scatter(
            x=chart.index,
            y=chart["Close"],
            name=symbol,
            line=dict(width=2, color="#2563EB"),
            row=1,
            col=1,
        )

        if "Volume" in chart:
            volume_colors = [
                "#16A34A" if close >= open_ else "#DC2626"
                for close, open_ in zip(
                    pd.to_numeric(chart["Close"], errors="coerce").fillna(0),
                    pd.to_numeric(chart.get("Open", chart["Close"]), errors="coerce").fillna(0),
                )
            ]
            fig.add_bar(
                x=chart.index,
                y=chart["Volume"],
                name="Volume",
                marker_color=volume_colors,
                row=2,
                col=1,
            )

        if {"MACD", "Signal", "MACD_Hist"}.issubset(chart.columns):
            macd_colors = ["#16A34A" if value >= 0 else "#DC2626" for value in chart["MACD_Hist"]]
            fig.add_bar(
                x=chart.index,
                y=chart["MACD_Hist"],
                name="Histogram",
                marker_color=macd_colors,
                row=3,
                col=1,
            )
            fig.add_scatter(
                x=chart.index,
                y=chart["MACD"],
                name="MACD",
                line=dict(width=1.7, color="#2563EB"),
                row=3,
                col=1,
            )
            fig.add_scatter(
                x=chart.index,
                y=chart["Signal"],
                name="Signal",
                line=dict(width=1.4, dash="dot", color="#F59E0B"),
                row=3,
                col=1,
            )

        fig.update_xaxes(range=[chart.index.min(), chart.index.max()], row=1, col=1)
        fig.update_xaxes(range=[chart.index.min(), chart.index.max()], row=2, col=1)
        fig.update_xaxes(range=[chart.index.min(), chart.index.max()], row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_layout(
            height=760,
            margin=dict(t=45, b=20, l=45, r=150),
            hovermode="x unified",
            legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
        )
        style_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)


def render_dcf(
    data: dict[str, Any],
    symbol: str | None = None,
    fmp_key: str | None = None,
    rapid_api_key: str | None = None,
) -> None:
    hints = estimate_dcf_assumptions(data)
    key_symbol = (symbol or str(data.get("company_name") or "valuation")).upper().replace(" ", "_")
    wacc_default = min(max(float(safe_num(hints.get("wacc_pct"), 10.0)), 1.0), 50.0)
    terminal_growth_default = min(max(float(safe_num(hints.get("terminal_growth_pct"), 2.5)), -5.0), 10.0)
    growth_1_default = min(max(float(safe_num(hints.get("growth_1_pct"), 12.0)), -50.0), 200.0)
    growth_2_default = min(max(float(safe_num(hints.get("growth_2_pct"), 7.0)), -50.0), 200.0)

    cols = st.columns(5)
    model = cols[0].selectbox("Model", ["standard", "margin"], key=f"{key_symbol}_dcf_model")
    wacc = cols[1].number_input(
        "WACC %",
        min_value=1.0,
        max_value=50.0,
        value=wacc_default,
        step=0.5,
        key=f"{key_symbol}_dcf_wacc",
    ) / 100
    terminal_growth = cols[2].number_input(
        "Terminal Growth %",
        min_value=-5.0,
        max_value=10.0,
        value=terminal_growth_default,
        step=0.1,
        key=f"{key_symbol}_dcf_terminal_growth",
    ) / 100
    growth_1 = cols[3].number_input(
        "Y1-5 Growth %",
        min_value=-50.0,
        max_value=200.0,
        value=growth_1_default,
        step=1.0,
        key=f"{key_symbol}_dcf_growth_1",
    ) / 100
    growth_2 = cols[4].number_input(
        "Y6-10 Growth %",
        min_value=-50.0,
        max_value=200.0,
        value=growth_2_default,
        step=1.0,
        key=f"{key_symbol}_dcf_growth_2",
    ) / 100
    st.caption(
        "Starter assumptions come from recent FCF/revenue growth, balance-sheet leverage, and ROCE; override them when filings or guidance tell a cleaner story."
    )
    target_margin = st.slider(
        "Target FCF Margin",
        min_value=0.01,
        max_value=0.80,
        value=0.25,
        step=0.01,
        key=f"{key_symbol}_dcf_target_margin",
    )

    try:
        valuation = calculate_dcf_valuation(
            data,
            model=model,
            wacc=wacc,
            terminal_growth=terminal_growth,
            fcf_growth_1=growth_1,
            fcf_growth_2=growth_2,
            revenue_growth_1=growth_1,
            revenue_growth_2=growth_2,
            target_fcf_margin=target_margin,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    dcf_cols = st.columns(4)
    dcf_cols[0].metric("Intrinsic Value", money(valuation.fair_value_per_share))
    dcf_cols[1].metric("Margin Of Safety", pct(valuation.margin_of_safety))
    dcf_cols[2].metric("Equity Value", format_compact_currency(valuation.equity_value))
    dcf_cols[3].metric("Terminal PV", format_compact_currency(valuation.present_value_terminal))

    bridge = pd.DataFrame(
        [
            {"Metric": "PV of 10Y FCF", "Amount": sum(valuation.present_value_fcf)},
            {"Metric": "PV of Terminal Value", "Amount": valuation.present_value_terminal},
            {"Metric": "Enterprise Value", "Amount": valuation.enterprise_value},
            {"Metric": "+ Cash", "Amount": safe_num(data.get("total_cash"))},
            {"Metric": "- Debt", "Amount": -safe_num(data.get("total_debt"))},
            {"Metric": "Equity Value", "Amount": valuation.equity_value},
            {"Metric": "Shares Outstanding", "Amount": valuation.shares_outstanding},
            {"Metric": "Intrinsic Value / Share", "Amount": valuation.fair_value_per_share},
        ]
    )
    render_dark_table(format_dcf_bridge_display(bridge))

    with st.expander("DCF Assumption Evidence", expanded=True):
        st.caption(
            "This panel helps source the WACC and growth knobs. It does not overwrite the model; it gives you material to defend or change the assumptions."
        )
        docs = pd.DataFrame()
        docs_key = f"{key_symbol}_dcf_docs_loaded"
        source_cols = st.columns([0.25, 0.75])
        if source_cols[0].button(
            "Load Filings / Transcripts",
            key=f"{key_symbol}_load_dcf_docs",
            disabled=not bool(symbol and fmp_key),
            use_container_width=True,
        ):
            st.session_state[docs_key] = True
        source_cols[1].caption(
            "Uses FMP document/transcript endpoints when your API plan exposes them. Empty results are normal for some plans."
        )
        if st.session_state.get(docs_key):
            with st.spinner("Loading DCF source material..."):
                docs = cached_dcf_source_documents(symbol or key_symbol, fmp_key)
        render_dark_table(build_dcf_assumption_evidence(data, docs), max_height_px=260)
        if st.session_state.get(docs_key):
            if docs.empty:
                st.info("No filings or transcripts were returned by the configured FMP endpoints for this ticker.")
            else:
                access_issues = docs[docs.get("Document", pd.Series(dtype="object")).astype(str).eq("Access issue")]
                if not access_issues.empty:
                    st.warning(
                        "FMP responded to at least one filing/transcript endpoint, but the configured key did not have access. "
                        "Check the FMP plan for SEC filings and earnings transcript permissions."
                    )
                render_dark_table(format_dcf_source_documents_display(docs), max_height_px=360)

        st.markdown("#### Management Tone From Earnings Call")
        tone_key = f"{key_symbol}_rapidapi_tone_loaded"
        tone_cols = st.columns([0.25, 0.75])
        if tone_cols[0].button(
            "Load Management Tone",
            key=f"{key_symbol}_load_rapidapi_tone",
            disabled=not bool(symbol and rapid_api_key),
            use_container_width=True,
        ):
            st.session_state[tone_key] = True
        tone_cols[1].caption(
            "Uses the subscribed RapidAPI Earnings Call Transcripts API. It isolates executive speaker segments when available."
        )
        if st.session_state.get(tone_key):
            with st.spinner("Loading latest earnings-call transcript and executive segments..."):
                bundle = cached_rapidapi_earnings_bundle(symbol or key_symbol, rapid_api_key)
            if bundle.get("status") != "ok":
                st.warning(str(bundle.get("message") or "RapidAPI transcript bundle could not be loaded."))
            else:
                tone_frame = build_management_tone_frame(bundle)
                tone_summary = management_tone_summary(bundle, tone_frame)
                tone_metrics = st.columns(4)
                tone_metrics[0].metric("Management Tone", str(tone_summary.get("tone") or "No data"))
                tone_metrics[1].metric("Tone Score", format_float(tone_summary.get("score"), 2))
                tone_metrics[2].metric("Exec Segments", str(tone_summary.get("executive_segments") or 0))
                tone_metrics[3].metric("Call Date", str(tone_summary.get("event_date") or "n/a")[:10])
                if tone_summary.get("title"):
                    st.caption(str(tone_summary["title"]))
                render_dark_table(format_management_tone_display(tone_frame), max_height_px=320)
                keywords = build_management_keyword_frame(bundle)
                if not keywords.empty:
                    render_dark_table(keywords, max_height_px=260)

    fig = go.Figure()
    years = [f"Y{index}" for index in range(1, 11)]
    fig.add_bar(x=years, y=valuation.future_fcf, name="Projected FCF")
    fig.add_scatter(x=years, y=valuation.present_value_fcf, name="Discounted PV", mode="lines+markers")
    fig.update_layout(height=320, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    style_dark_plotly(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_price_targets(symbol: str, fmp_key: str | None) -> None:
    targets = cached_price_targets(symbol, fmp_key)
    cols = st.columns(4)
    cols[0].metric("Consensus", money(targets.get("targetConsensus")))
    cols[1].metric("High", money(targets.get("targetHigh")))
    cols[2].metric("Low", money(targets.get("targetLow")))
    cols[3].metric("Analysts", str(targets.get("numberOfAnalystOpinions") or "missing"))
    if not targets:
        st.info("Price target data is unavailable from FMP and Yahoo for this symbol right now.")
        return

    source = targets.get("source")
    if source:
        st.caption(f"Source: {source}")

    target_rows = pd.DataFrame(targets.get("targetRows") or [])
    if target_rows.empty:
        fallback_rows = [
            {"Published Date": "summary", "Firm": "Consensus range", "Rating": "Low", "Target": targets.get("targetLow")},
            {"Published Date": "summary", "Firm": "Consensus range", "Rating": "Consensus", "Target": targets.get("targetConsensus")},
            {"Published Date": "summary", "Firm": "Consensus range", "Rating": "High", "Target": targets.get("targetHigh")},
        ]
        target_rows = pd.DataFrame(fallback_rows)
        target_rows["Target"] = pd.to_numeric(target_rows["Target"], errors="coerce")
        target_rows = target_rows.dropna(subset=["Target"])
        if not target_rows.empty:
            st.info("Detailed analyst-level target rows are unavailable, so this chart shows the available consensus range.")

    if target_rows.empty or "Target" not in target_rows:
        return

    target_rows = target_rows.copy()
    target_rows["Target"] = pd.to_numeric(target_rows["Target"], errors="coerce")
    target_rows = target_rows.dropna(subset=["Target"])
    target_rows["Published Date"] = target_rows.get("Published Date", pd.Series(dtype="object")).replace("", pd.NA)
    parsed_dates = pd.to_datetime(target_rows["Published Date"], errors="coerce")
    target_rows["Plot Date"] = parsed_dates
    if target_rows["Plot Date"].notna().any():
        target_rows = target_rows.sort_values("Plot Date")
        x_values = target_rows["Plot Date"]
    else:
        target_rows = target_rows.reset_index(drop=True)
        x_values = target_rows.index + 1

    for column in ("Firm", "Analyst", "Rating", "Source", "Title"):
        if column not in target_rows:
            target_rows[column] = ""

    scatter = go.Figure()
    scatter.add_trace(
        go.Scatter(
            x=x_values,
            y=target_rows["Target"],
            mode="markers",
            marker=dict(size=11),
            text=target_rows["Firm"],
            customdata=target_rows[["Analyst", "Rating", "Source", "Title"]].fillna(""),
            name="Analyst Targets",
            hovertemplate=(
                "<b>%{text}</b><br>Target: $%{y:.2f}"
                "<br>%{customdata[0]}<br>%{customdata[1]}"
                "<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    consensus = safe_num(targets.get("targetConsensus"))
    high = safe_num(targets.get("targetHigh"))
    low = safe_num(targets.get("targetLow"))
    if consensus:
        scatter.add_hline(y=consensus, line_dash="dash", line_color="#16A34A", annotation_text=f"Consensus ${consensus:.2f}")
    if high:
        scatter.add_hline(y=high, line_dash="dot", line_color="#D97706", annotation_text=f"High ${high:.2f}")
    if low:
        scatter.add_hline(y=low, line_dash="dot", line_color="#DC2626", annotation_text=f"Low ${low:.2f}")
    scatter.update_layout(height=360, margin=dict(t=20, b=20, l=10, r=10), hovermode="closest")
    style_dark_plotly(scatter, hovermode="closest")
    st.plotly_chart(scatter, use_container_width=True, theme=None)

    display_columns = [
        column
        for column in ["Published Date", "Firm", "Analyst", "Rating", "Target", "Source", "Title", "URL"]
        if column in target_rows
    ]
    table = target_rows[display_columns].copy()
    if "Plot Date" in table:
        table = table.drop(columns=["Plot Date"])
    render_dark_table(table, max_height_px=420)


def render_peers(symbol: str, fmp_key: str | None) -> None:
    peer_data = cached_peer_comparison(symbol, fmp_key)
    if peer_data.peer_symbols:
        st.caption(
            "Peer universe from FMP stock-peers, independent of your watchlist: "
            + ", ".join(peer_data.peer_symbols)
        )
    if peer_data.error:
        st.info(peer_data.error)
        return
    if not peer_data.metrics.empty:
        cols = [column for column in VALUATION_MULTIPLE_COLUMNS if column in peer_data.metrics.columns]
        if cols:
            config = {
                column: st.column_config.NumberColumn(
                    label=VALUATION_MULTIPLE_COLUMNS[column][0],
                    help=VALUATION_MULTIPLE_COLUMNS[column][1],
                    format="%.2f",
                )
                for column in cols
            }
            config["symbol"] = st.column_config.TextColumn("Symbol")
            display = peer_data.metrics[cols].reset_index()
            render_dark_table(display, max_height_px=360)
    if not peer_data.ratios.empty:
        category = st.selectbox("Ratio category", list(RATIO_CATEGORIES.keys()))
        selected = RATIO_CATEGORIES[category]
        cols = [column for column in selected if column in peer_data.ratios.columns]
        if cols:
            config = {
                column: st.column_config.NumberColumn(label=selected[column][0], help=selected[column][1], format="%.2f")
                for column in cols
            }
            config["symbol"] = st.column_config.TextColumn("Symbol")
            display = peer_data.ratios[cols].reset_index()
            render_dark_table(display, max_height_px=360)


def render_supply_chain_tab(symbol: str, data: dict[str, Any], settings: Any) -> None:
    peer_data = cached_peer_comparison(symbol, settings.fmp_api_key)
    peers = [peer for peer in peer_data.peer_symbols if peer != symbol.upper()]
    saved_rows = load_supply_chain_rows(symbol)
    rows = saved_rows or default_supply_chain_rows(symbol, data, peers[:6])
    path = supply_chain_path(symbol)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Relationships", str(len(rows)))
    metric_cols[1].metric("FMP Peers", str(len(peers)))
    metric_cols[2].metric("LLM Key", present(getattr(settings, "llm_api_key", None)))
    metric_cols[3].metric("Saved", "yes" if saved_rows else "not yet")
    st.caption(f"Supply-chain rows path: `{display_path(path)}`")

    frame = pd.DataFrame(rows)
    expected_columns = ["From", "To", "Relationship", "Layer", "Weight", "Evidence", "Source"]
    for column in expected_columns:
        if column not in frame:
            frame[column] = ""
    frame = frame[expected_columns]

    render_supply_chain_graph(frame)

    st.subheader("Relationship Evidence")
    edited = st.data_editor(
        frame,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Layer": st.column_config.SelectboxColumn(
                "Layer",
                options=["Upstream", "Downstream", "Competitive Set", "Other"],
            ),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.1, max_value=10.0, step=0.1),
            "Evidence": st.column_config.TextColumn("Evidence", width="large"),
        },
        key=f"supply_chain_editor_{symbol}",
    )
    actions = st.columns([1, 1, 3])
    if actions[0].button(T("save_rows"), use_container_width=True):
        clean = edited.fillna("").to_dict("records")
        saved_path = save_supply_chain_rows(symbol, clean)
        st.success(f"Saved to {display_path(saved_path)}")
    if actions[1].button(T("reset_starter"), use_container_width=True):
        if path.exists():
            path.unlink()
        st.rerun()

    with st.expander("LLM extraction slot"):
        st.write(
            "This is where we can later paste a 10-K excerpt, annual report section, supplier disclosure, "
            "or news article and ask the configured LLM to extract structured relationship rows into the table above."
        )
        st.code(
            json.dumps(
                {
                    "task": "Extract supply-chain relationships for a public company.",
                    "ticker": symbol.upper(),
                    "output_schema": {
                        "From": "source company, segment, supplier group, or customer group",
                        "To": "target company, segment, supplier group, or customer group",
                        "Relationship": "short relationship verb",
                        "Layer": "Upstream | Downstream | Competitive Set | Other",
                        "Weight": "1-10 confidence/materiality score",
                        "Evidence": "short quoted or paraphrased evidence",
                        "Source": "document/source name",
                    },
                },
                indent=2,
            ),
            language="json",
        )


def render_stock_valuation_tab(settings: Any) -> None:
    st.subheader(T("stock_valuation"))
    symbol = st.text_input(T("valuation_ticker"), value=st.session_state.get("workbench_symbol", "AAPL")).upper().strip()
    if st.button(T("load_valuation"), type="primary", disabled=not symbol):
        st.session_state["valuation_symbol"] = symbol
        st.session_state["workbench_symbol"] = symbol

    loaded = st.session_state.get("valuation_symbol")
    if not loaded:
        st.info(T("choose_ticker_load"))
        return

    with st.spinner(T("loading_fundamentals", symbol=loaded)):
        data, financials, historicals, mandates = cached_fundamental_data(loaded, settings.fmp_api_key)
    if not data or safe_num(data.get("price")) <= 0:
        st.warning(T("valuation_snapshot_missing"))
        return

    render_stock_snapshot(loaded, data)
    valuation_tab, targets_tab, peers_tab, supply_chain_tab, checklist_tab = st.tabs(
        ops_tabs(OPS_LANG, "valuation_tabs")
    )
    with valuation_tab:
        render_dcf(data, loaded, settings.fmp_api_key, settings.rapid_api_key)
    with targets_tab:
        render_price_targets(loaded, settings.fmp_api_key)
    with peers_tab:
        render_peers(loaded, settings.fmp_api_key)
    with supply_chain_tab:
        render_supply_chain_tab(loaded, data, settings)
    with checklist_tab:
        if not mandates:
            st.info(T("checklist_unavailable"))
        else:
            score = sum(1 for passed in mandates.values() if passed)
            st.progress(score / max(len(mandates), 1), text=f"{score}/{len(mandates)} checks passed")
            render_dark_table(
                pd.DataFrame({"Criterion": list(mandates), "Passed": list(mandates.values())}),
            )


def scanner_display(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No candidates passed the current filters.")
        return
    formatted = format_scanner_frame(frame)
    display = formatted.drop(columns=["Legs"], errors="ignore")
    display = format_options_scanner_display(display)
    render_dark_table(display, max_height_px=480)


def candidate_label(row: pd.Series) -> str:
    strategy = str(row.get("Strategy", "Candidate"))
    expiry = str(row.get("Expiry", ""))
    structure = str(row.get("Structure", row.get("Strike", "")))
    debit_credit = money(row.get("Debit/Credit"))
    return f"{strategy} | {expiry} | {structure} | {debit_credit}"


def render_proposal_controls(frame: pd.DataFrame, *, underlying: str, key_prefix: str) -> None:
    if frame.empty or "Legs" not in frame.columns:
        return
    with st.expander(T("write_draft_proposal"), expanded=False):
        st.caption(T("proposal_slot_caption"))
        contracts = st.number_input(T("contracts_per_leg"), min_value=1, max_value=10, value=1, step=1, key=f"{key_prefix}_contracts")
        proposal_frame = frame.reset_index(drop=True)
        labels = [f"{index + 1}. {candidate_label(row)}" for index, (_, row) in enumerate(proposal_frame.iterrows())]
        selected = st.selectbox(T("candidate"), labels, key=f"{key_prefix}_candidate")
        selected_index = labels.index(selected)
        if st.button(T("write_draft_button"), key=f"{key_prefix}_write"):
            try:
                result = write_option_trade_proposal_from_candidate(
                    proposal_frame.iloc[selected_index].to_dict(),
                    underlying=underlying,
                    contracts=float(contracts),
                )
            except Exception as exc:
                st.error(T("could_not_write_proposal", error=exc))
                return
            st.success(T("draft_proposal_written", path=display_path(result.written_path)))
            st.caption(T("proposal_id", proposal_id=result.proposal.proposal_id))


def render_directional_lens(lens: DirectionalLensResult, *, chain_source: str) -> None:
    st.caption(
        "This is a cross-check, not an execution signal. It blends price trend, RSI, "
        "FMP sentiment when available, analyst posture, and option skew."
    )
    if lens.horizon_frame.empty:
        st.info("Directional lens could not build a usable signal frame.")
        return

    primary_direction = lens.summary.get("primary_direction", "Neutral")
    primary_score = safe_num(lens.summary.get("primary_score"), 0.0)
    primary_confidence = safe_num(lens.summary.get("primary_confidence"), 0.0)
    one_day = lens.horizon_frame[lens.horizon_frame["Horizon"].eq("1D")]
    one_month = lens.horizon_frame[lens.horizon_frame["Horizon"].eq("1M")]

    metrics = st.columns(5)
    metrics[0].metric("Primary Read", primary_direction)
    metrics[1].metric("1W Score", format_float(primary_score, 2))
    metrics[2].metric("Confidence", format_percent_cell(primary_confidence, 1))
    metrics[3].metric("1D Direction", str(one_day.iloc[0]["Direction"]) if not one_day.empty else "Neutral")
    metrics[4].metric("1M Direction", str(one_month.iloc[0]["Direction"]) if not one_month.empty else "Neutral")
    st.caption(f"Options skew source: {chain_source}. FMP sentiment rows: {len(lens.sentiment_frame)}.")

    frame = lens.horizon_frame.copy()
    colors = frame["Score"].map(
        lambda value: "#22c55e" if safe_num(value) >= 0.20 else "#ef4444" if safe_num(value) <= -0.20 else "#f59e0b"
    )
    fig = go.Figure()
    fig.add_bar(
        x=frame["Horizon"],
        y=frame["Score"],
        marker_color=colors,
        text=frame["Direction"],
        textposition="outside",
        hovertemplate="%{x}<br>Score: %{y:.2f}<br>%{text}<extra></extra>",
    )
    fig.add_hline(y=0, line_color="rgba(148, 163, 184, 0.55)", line_width=1)
    fig.update_yaxes(range=[-1, 1], title="Directional Score")
    fig.update_layout(height=330, margin=dict(t=25, b=30, l=20, r=20), showlegend=False)
    style_dark_plotly(fig, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    horizon_tab, contribution_tab, sentiment_tab = st.tabs(ops_tabs(OPS_LANG, "directional_lens_tabs"))
    with horizon_tab:
        render_dark_table(format_directional_horizon_display(lens.horizon_frame), max_height_px=360)
    with contribution_tab:
        render_dark_table(format_directional_contribution_display(lens.contribution_frame), max_height_px=420)
    with sentiment_tab:
        if lens.sentiment_frame.empty:
            st.info("No FMP sentiment rows were available for this symbol. The lens is using price, RSI, analyst, and options signals.")
        else:
            render_dark_table(format_sentiment_display(lens.sentiment_frame), max_height_px=420)


def render_keyword_cloud(keyword_frame: pd.DataFrame) -> None:
    if keyword_frame.empty:
        st.info("No keywords extracted yet.")
        return
    frame = keyword_frame.head(36).copy()
    counts = pd.to_numeric(frame["Count"], errors="coerce").fillna(0)
    max_count = max(float(counts.max()), 1.0)
    color_map = {"Bullish": "#22c55e", "Bearish": "#fb7185", "Neutral": "#38bdf8"}
    fig = go.Figure()
    for index, row in frame.reset_index(drop=True).iterrows():
        angle = float(index) * 2.399963229728653
        radius = 0.28 + float(np.sqrt(index + 1))
        count = safe_num(row.get("Count"), 0.0)
        tone = str(row.get("Tone") or "Neutral")
        fig.add_scatter(
            x=[np.cos(angle) * radius],
            y=[np.sin(angle) * radius],
            mode="text",
            text=[str(row.get("Keyword") or "")],
            textfont=dict(
                size=14 + 24 * (count / max_count),
                color=color_map.get(tone, "#38bdf8"),
            ),
            hovertext=[f"{row.get('Keyword')}<br>Count: {int(count)}<br>Tone: {tone}"],
            hoverinfo="text",
            showlegend=False,
        )
    fig.update_layout(height=360, margin=dict(t=12, b=12, l=12, r=12), showlegend=False)
    style_dark_plotly(fig, hovermode=False)
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_news_nlp_section(
    nlp: NewsNLPResult,
    *,
    cache_state: dict[str, Any],
    provider_frame: pd.DataFrame,
) -> None:
    metrics = st.columns(5)
    metrics[0].metric("News Tone", str(nlp.summary.get("sentiment_label") or "No data"))
    metrics[1].metric("NLP Score", format_float(nlp.summary.get("sentiment_score"), 2))
    metrics[2].metric("Articles", str(nlp.summary.get("article_count") or 0))
    metrics[3].metric("Cache", str(cache_state.get("state") or "missing"))
    metrics[4].metric("Age", format_float(cache_state.get("age_hours"), 1, "h") if cache_state.get("age_hours") is not None else "missing")
    if nlp.summary.get("top_topics"):
        st.caption(f"Top topics: {nlp.summary['top_topics']}")
    if nlp.summary.get("top_keywords"):
        st.caption(f"Top keywords: {nlp.summary['top_keywords']}")

    if nlp.article_frame.empty:
        st.info("No FMP articles were available. Once FMP returns news rows, this section will score tone, topics, and keywords locally.")
        return

    catalyst_board = build_news_catalyst_board(nlp.article_frame)
    evidence_frame = build_news_evidence_frame(nlp.article_frame)

    col_left, col_right = st.columns([1.1, 0.9])
    with col_left:
        timeline = nlp.article_frame.copy()
        timeline["Date"] = pd.to_datetime(timeline["Published"], errors="coerce").dt.date
        timeline["Score"] = pd.to_numeric(timeline["Score"], errors="coerce")
        timeline = timeline.dropna(subset=["Date", "Score"])
        if not timeline.empty:
            grouped = timeline.groupby("Date", as_index=False).agg(Score=("Score", "mean"), Articles=("Score", "size"))
            fig = go.Figure()
            fig.add_scatter(
                x=grouped["Date"],
                y=grouped["Score"],
                mode="lines+markers",
                name="Sentiment",
                line=dict(color="#38bdf8", width=2),
                customdata=grouped[["Articles"]],
                hovertemplate="%{x}<br>Score: %{y:.2f}<br>Articles: %{customdata[0]}<extra></extra>",
            )
            fig.add_hline(y=0, line_color="rgba(148, 163, 184, 0.55)", line_width=1)
            fig.update_yaxes(range=[-1, 1], title="Tone")
            fig.update_layout(height=320, margin=dict(t=25, b=30, l=20, r=20))
            style_dark_plotly(fig, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            st.info("Article dates are not parseable enough for a sentiment timeline.")
    with col_right:
        if not nlp.keyword_frame.empty:
            keywords = nlp.keyword_frame.head(15).sort_values("Count")
            color_map = {"Bullish": "#22c55e", "Bearish": "#ef4444", "Neutral": "#38bdf8"}
            fig = go.Figure()
            fig.add_bar(
                x=keywords["Count"],
                y=keywords["Keyword"],
                orientation="h",
                marker_color=keywords["Tone"].map(color_map).fillna("#38bdf8"),
                hovertemplate="%{y}<br>Count: %{x}<extra></extra>",
            )
            fig.update_layout(height=320, margin=dict(t=25, b=30, l=20, r=20), showlegend=False)
            style_dark_plotly(fig)
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            st.info("No keywords extracted.")

    catalyst_tab, evidence_tab, cloud_tab, topic_tab, provider_tab, article_tab = st.tabs(
        ["Catalyst Board", "Evidence", "Keyword Cloud", "Topics", "Provider Readiness", "Articles"]
    )
    with catalyst_tab:
        render_dark_table(format_news_catalyst_display(catalyst_board), max_height_px=360)
    with evidence_tab:
        render_dark_table(format_news_evidence_display(evidence_frame), max_height_px=420)
    with cloud_tab:
        render_keyword_cloud(nlp.keyword_frame)
    with topic_tab:
        render_dark_table(format_news_topic_display(nlp.topic_frame), max_height_px=320)
    with provider_tab:
        render_dark_table(format_provider_status_display(provider_frame), max_height_px=360)
    with article_tab:
        render_dark_table(format_news_article_display(nlp.article_frame), max_height_px=520)


def render_ai_outlook_section(
    *,
    symbol: str,
    data: dict[str, Any],
    articles: pd.DataFrame,
    settings: Any,
) -> None:
    st.caption(
        "Synthesizes transcript, news, and valuation context into short-term, mid-term, and long-term evidence. "
        "The result is cached locally; it does not overwrite valuation inputs."
    )
    llm_api_key = getattr(settings, "llm_api_key", None)
    enabled = bool(getattr(settings, "llm_evidence_enabled", False))
    status_cols = st.columns(5)
    status_cols[0].metric("Provider", str(getattr(settings, "llm_provider", "missing")))
    status_cols[1].metric("Model", str(getattr(settings, "llm_model", "missing")))
    status_cols[2].metric("LLM Key", present(llm_api_key))
    status_cols[3].metric("RapidAPI", present(settings.rapid_api_key))
    status_cols[4].metric("Cache", "local")
    st.caption(f"LLM cache path: `{display_path(DEFAULT_LLM_EVIDENCE_DB_PATH)}`")

    if not enabled:
        st.info("LLM evidence is disabled. Set `LLM_EVIDENCE_ENABLED=true` if you want this panel active.")
    if not llm_api_key:
        st.warning("No LLM API key is configured. For GLM/Z.ai, set `ZAI_API_KEY` in `.env`.")

    key_prefix = f"{symbol.upper()}_ai_outlook"
    controls = st.columns([0.22, 0.22, 0.56])
    force_refresh = controls[1].checkbox("Force refresh", value=False, key=f"{key_prefix}_force")
    if controls[0].button(
        "Run AI Outlook",
        type="primary",
        disabled=not (enabled and llm_api_key),
        use_container_width=True,
        key=f"{key_prefix}_run",
    ):
        with st.spinner("Loading transcript packet and asking the configured LLM..."):
            transcript_bundle = (
                cached_rapidapi_earnings_bundle(symbol, settings.rapid_api_key)
                if settings.rapid_api_key
                else {}
            )
            result = analyze_company_outlook(
                symbol=symbol,
                api_key=llm_api_key,
                provider=getattr(settings, "llm_provider", "zai"),
                base_url=getattr(settings, "llm_base_url", "https://api.z.ai/api/paas/v4"),
                model=getattr(settings, "llm_model", "glm-5.2"),
                transcript_bundle=transcript_bundle,
                news_articles=articles,
                valuation_data=data,
                max_age_hours=getattr(settings, "llm_cache_max_age_hours", 168.0),
                timeout_seconds=int(getattr(settings, "llm_timeout_seconds", 60) or 60),
                force_refresh=force_refresh,
            )
        st.session_state[key_prefix] = result

    controls[2].caption(
        "Use force refresh only when you intentionally want a new model call for the same source packet."
    )
    result = st.session_state.get(key_prefix)
    if not result:
        st.info("Run the synthesis after loading a ticker. Cached results will return without another model call.")
        return
    if result.get("status") == "error":
        st.error(str(result.get("message") or "LLM synthesis failed."))
        return

    analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else {}
    badge_cols = st.columns(4)
    badge_cols[0].metric("Status", str(result.get("status") or "missing"))
    badge_cols[1].metric("Provider", str(result.get("provider") or "missing"))
    badge_cols[2].metric("Model", str(result.get("model") or "missing"))
    badge_cols[3].metric("Created", str(result.get("created_at") or "cached")[:19])
    if analysis.get("executive_summary"):
        st.info(str(analysis["executive_summary"]))

    horizon_tab, dcf_tab, options_tab, risks_tab = st.tabs(
        ["Outlook", "DCF Clues", "Options Implications", "Risks & Limits"]
    )
    with horizon_tab:
        render_dark_table(format_ai_horizon_frame(analysis), max_height_px=420)
    with dcf_tab:
        render_dark_table(
            format_ai_key_value_frame(analysis.get("dcf_assumption_clues"), key_label="DCF Input"),
            max_height_px=360,
        )
    with options_tab:
        render_dark_table(
            format_ai_key_value_frame(analysis.get("options_implications"), key_label="Options Lens"),
            max_height_px=360,
        )
    with risks_tab:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Key Risks")
            render_dark_table(format_ai_list_frame(analysis.get("key_risks"), column="Risk"), max_height_px=300)
        with col_b:
            st.markdown("#### Source Limits")
            render_dark_table(
                format_ai_list_frame(analysis.get("source_limits"), column="Limit"),
                max_height_px=300,
            )


def opportunity_action_bucket(
    *,
    spot: float,
    price_targets: dict[str, Any],
    forecast_vol: float,
    market_iv: float,
    directional_lens: DirectionalLensResult,
    news_nlp: NewsNLPResult,
) -> tuple[str, str]:
    target = safe_num(price_targets.get("targetConsensus"), 0.0)
    target_upside = (target / spot) - 1 if spot > 0 and target > 0 else 0.0
    direction = str(directional_lens.summary.get("primary_direction") or "Neutral")
    direction_score = safe_num(directional_lens.summary.get("primary_score"), 0.0)
    news_score = safe_num(news_nlp.summary.get("sentiment_score"), 0.0)
    vrp = market_iv - forecast_vol

    if target_upside > 0.15 and direction != "Bearish" and news_score > -0.15:
        return "Shares / staged entry", f"Analyst upside is {target_upside:.1%}; direction/news do not veto owning shares."
    if abs(direction_score) >= 0.30 and vrp < -0.03:
        return "Long options / convexity", f"{direction} read with forecast vol above market IV by {abs(vrp):.1%}."
    if abs(direction_score) >= 0.30 and vrp >= -0.03:
        return "Defined-risk spread", f"{direction} read, but vol is not obviously cheap; prefer controlled premium."
    if abs(direction_score) < 0.20 and vrp > 0.04:
        return "Income / range structures", f"Directional read is neutral while market IV is rich by {vrp:.1%}."
    return "Watch / wait", "No strong share-or-options edge from the combined snapshot yet."


def render_opportunity_pressure_chart(lens_frame: pd.DataFrame) -> None:
    if lens_frame.empty:
        st.info("No opportunity lenses are available yet.")
        return
    frame = lens_frame.copy()
    frame["Score"] = pd.to_numeric(frame["Score"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("Score")
    colors = frame["Score"].map(
        lambda value: "#22c55e" if value > 0.20 else "#fb7185" if value < -0.20 else "#f59e0b"
    )
    fig = go.Figure()
    fig.add_bar(
        x=frame["Score"],
        y=frame["Lens"],
        orientation="h",
        marker_color=colors,
        customdata=frame[["Bias", "Detail"]],
        hovertemplate="%{y}<br>Score: %{x:.2f}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    )
    fig.add_vline(x=0, line_color="rgba(148, 163, 184, 0.45)", line_width=1)
    fig.update_xaxes(range=[-1, 1], title="Pressure")
    fig.update_layout(height=340, margin=dict(t=18, b=28, l=20, r=20), showlegend=False)
    style_dark_plotly(fig, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_opportunity_decision_cockpit(
    *,
    spot: float,
    target_consensus: float,
    direction: str,
    direction_score: float,
    news_label: str,
    news_score: float,
    forecast_vol: float,
    market_iv: float,
    rsi_14: float,
    expiration_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.subheader("Decision Cockpit")
    st.caption("A discretionary read: where the pressure is coming from, and which vehicle deserves attention first.")
    target_upside = (target_consensus / spot) - 1.0 if spot > 0 and target_consensus > 0 else None
    lens_frame = build_opportunity_lens_frame(
        spot=spot,
        target_consensus=target_consensus,
        direction=direction,
        direction_score=direction_score,
        news_label=news_label,
        news_score=news_score,
        forecast_vol=forecast_vol,
        market_iv=market_iv,
        rsi_14=rsi_14,
        expiration_count=expiration_count,
    )
    route_frame = build_vehicle_route_frame(
        target_upside=target_upside,
        direction=direction,
        direction_score=direction_score,
        news_score=news_score,
        forecast_vol=forecast_vol,
        market_iv=market_iv,
    )
    left, right = st.columns([1.05, 1.25])
    with left:
        render_opportunity_pressure_chart(lens_frame)
    with right:
        render_dark_table(format_vehicle_route_display(route_frame), max_height_px=340)
    with st.expander("Lens Details", expanded=False):
        render_dark_table(format_opportunity_lens_display(lens_frame), max_height_px=320)
    return lens_frame, route_frame


def top_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return frame.head(1).copy()


def build_strategy_fit_candidates(
    *,
    symbol: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry: str,
    expirations: list[str],
    budget: float,
    hold_days: int,
    forecast_vol: float,
    settings: Any,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    rows.append(top_candidate(scan_long_options(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_long_options(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_cash_secured_puts(puts, spot=spot, expiry=expiry, max_collateral=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_vertical_spreads(calls, spot=spot, expiry=expiry, spread_type="bull_call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_vertical_spreads(puts, spot=spot, expiry=expiry, spread_type="bear_put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_iron_condors(calls, puts, spot=spot, expiry=expiry, max_risk=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_call_butterflies(calls, spot=spot, expiry=expiry, budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_ratio_spreads(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_ratio_spreads(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_backspreads(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    rows.append(top_candidate(scan_backspreads(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)))
    far_expiry = far_calendar_expiry(expirations, expiry)
    if far_expiry is not None:
        far_calls, _, _ = cached_option_chain(symbol, far_expiry, settings)
        rows.append(top_candidate(scan_calendar_spreads(calls, far_calls, spot=spot, near_expiry=expiry, far_expiry=far_expiry, budget=budget, forecast_vol=forecast_vol)))

    candidates = [frame for frame in rows if not frame.empty]
    if not candidates:
        return pd.DataFrame()
    combined = pd.concat(candidates, ignore_index=True)
    combined = combined.sort_values(["EV", "PoP"], ascending=False).reset_index(drop=True)
    return combined


def render_strategy_fit(
    symbol: str,
    history: pd.DataFrame,
    expirations: list[str],
    spot: float,
    forecast_vol: float,
    hold_days: int,
    budget: float,
    settings: Any,
    directional_lens: DirectionalLensResult | None = None,
) -> None:
    if not expirations:
        st.info("No listed options expirations found.")
        return
    expiry = choose_expiration(expirations, hold_days)
    calls, puts, chain_source = cached_option_chain(symbol, expiry, settings) if expiry else (pd.DataFrame(), pd.DataFrame(), "n/a")
    market_iv = estimate_atm_market_iv(calls, puts, spot, forecast_vol or 0.20)
    snapshot = volatility_snapshot(history)
    candidates = build_strategy_fit_candidates(
        symbol=symbol,
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=expiry or "",
        expirations=expirations,
        budget=budget,
        hold_days=hold_days,
        forecast_vol=forecast_vol or 0.20,
        settings=settings,
    )
    cols = st.columns(6)
    cols[0].metric("ATM IV", pct(market_iv))
    cols[1].metric("Forecast Vol", pct(forecast_vol or 0.20))
    cols[2].metric("VRP", pct(market_iv - (forecast_vol or 0.20)))
    cols[3].metric("RSI 14", f"{snapshot.rsi_14:.1f}")
    cols[4].metric("Reference Expiry", expiry or "n/a")
    cols[5].metric("Chain Source", chain_source)
    if candidates.empty:
        st.info("No contract-level candidates passed the current filters.")
    else:
        display = format_scanner_frame(candidates).drop(columns=["Legs"], errors="ignore")
        display = add_strategy_direction_columns(display, directional_lens, horizon="1W")
        if "Max Loss" in display and "EV" in display:
            max_loss = pd.to_numeric(display["Max Loss"], errors="coerce").abs()
            display["EV / Max Loss"] = pd.to_numeric(display["EV"], errors="coerce") / max_loss.replace(0, pd.NA)
        display_cols = [
            column
            for column in [
                "Strategy",
                "Payoff Direction",
                "Model Direction",
                "Alignment",
                "Expiry",
                "Structure",
                "PoP",
                "EV",
                "EV / Max Loss",
                "VaR 95",
                "Max Profit",
                "Max Loss",
                "Debit/Credit",
                "Edge",
                "IV Edge",
                "Market IV",
            ]
            if column in display.columns
        ]
        render_dark_table(format_options_scanner_display(display[display_cols].head(12)), max_height_px=420)


def render_options_playbook(
    *,
    symbol: str,
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry: str,
    expirations: list[str],
    budget: float,
    hold_days: int,
    forecast_vol: float,
    market_iv: float,
    chain_source: str,
    settings: Any,
    route_frame: pd.DataFrame,
) -> None:
    st.subheader("Options Playbook")
    st.caption("A compact triage table that connects the decision cockpit to actual option structures.")
    if not expiry:
        st.info("No reference expiration is available for this ticker.")
        return
    candidates = build_strategy_fit_candidates(
        symbol=symbol,
        calls=calls,
        puts=puts,
        spot=spot,
        expiry=expiry,
        expirations=expirations,
        budget=budget,
        hold_days=hold_days,
        forecast_vol=forecast_vol or 0.20,
        settings=settings,
    )
    playbook = build_options_playbook_frame(
        route_frame=route_frame,
        candidates=candidates,
        forecast_vol=forecast_vol or 0.20,
        market_iv=market_iv,
        reference_expiry=expiry,
        chain_source=chain_source,
    )
    cols = st.columns(5)
    cols[0].metric("Reference Expiry", expiry)
    cols[1].metric("ATM IV", pct(market_iv))
    cols[2].metric("Forecast Vol", pct(forecast_vol or 0.20))
    cols[3].metric("Candidate Rows", str(len(candidates)))
    cols[4].metric("Chain Source", chain_source)
    render_dark_table(format_options_playbook_display(playbook), max_height_px=420)
    with st.expander("Top Raw Candidates", expanded=False):
        if candidates.empty:
            st.info("No scanner candidates passed the current filters.")
        else:
            display = format_scanner_frame(candidates).drop(columns=["Legs"], errors="ignore")
            render_dark_table(format_options_scanner_display(display.head(12)), max_height_px=420)


def render_scanner(symbol: str, spot: float, expirations: list[str], budget: float, hold_days: int, forecast_vol: float, settings: Any) -> None:
    if not expirations:
        st.warning("No listed options expirations found.")
        return
    default_expiry = choose_expiration(expirations, hold_days)
    default_index = expirations.index(default_expiry) if default_expiry in expirations else 0
    expiry = st.selectbox("Expiration", expirations, index=default_index)
    calls, puts, chain_source = cached_option_chain(symbol, expiry, settings)
    st.caption(f"Option chain source for this expiration: {chain_source}")

    scans = st.tabs(ops_tabs(OPS_LANG, "option_scanner_tabs"))
    with scans[0]:
        frame = scan_long_options(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_long_calls")
    with scans[1]:
        frame = scan_long_options(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_long_puts")
    with scans[2]:
        frame = scan_cash_secured_puts(puts, spot=spot, expiry=expiry, max_collateral=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_short_puts")
    with scans[3]:
        st.subheader("Bull Calls")
        frame = scan_vertical_spreads(calls, spot=spot, expiry=expiry, spread_type="bull_call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_bull_calls")
        st.subheader("Bear Puts")
        frame = scan_vertical_spreads(puts, spot=spot, expiry=expiry, spread_type="bear_put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_bear_puts")
    with scans[4]:
        far_expiry = far_calendar_expiry(expirations, expiry)
        if far_expiry is None:
            st.info("No farther expiration found for a calendar spread.")
        else:
            far_calls, _, far_source = cached_option_chain(symbol, far_expiry, settings)
            st.caption(f"Far expiration source: {far_source}")
            frame = scan_calendar_spreads(calls, far_calls, spot=spot, near_expiry=expiry, far_expiry=far_expiry, budget=budget, forecast_vol=forecast_vol)
            scanner_display(frame)
            render_proposal_controls(frame, underlying=symbol, key_prefix="work_calendars")
    with scans[5]:
        frame = scan_iron_condors(calls, puts, spot=spot, expiry=expiry, max_risk=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_condors")
    with scans[6]:
        frame = scan_call_butterflies(calls, spot=spot, expiry=expiry, budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_butterflies")
    with scans[7]:
        st.subheader("Call Ratios")
        frame = scan_ratio_spreads(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_call_ratios")
        st.subheader("Put Ratios")
        frame = scan_ratio_spreads(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_put_ratios")
    with scans[8]:
        st.subheader("Call Backspreads")
        frame = scan_backspreads(calls, spot=spot, expiry=expiry, option_type="call", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_call_backspreads")
        st.subheader("Put Backspreads")
        frame = scan_backspreads(puts, spot=spot, expiry=expiry, option_type="put", budget=budget, days_to_hold=hold_days, forecast_vol=forecast_vol)
        scanner_display(frame)
        render_proposal_controls(frame, underlying=symbol, key_prefix="work_put_backspreads")


def render_options_tab(settings: Any) -> None:
    st.subheader("Options Scanner")
    controls = st.columns(4)
    symbol = controls[0].text_input("Options ticker", value=st.session_state.get("workbench_symbol", "QQQ")).upper().strip()
    budget = controls[1].number_input("Budget / Max Risk", min_value=100.0, value=1000.0, step=100.0)
    hold_days = controls[2].number_input("Target Hold Days", min_value=1, max_value=365, value=30, step=1)
    if controls[3].button("Load Options", type="primary", disabled=not symbol):
        st.session_state["options_symbol"] = symbol
        st.session_state["options_budget"] = float(budget)
        st.session_state["options_hold_days"] = int(hold_days)
        st.session_state["workbench_symbol"] = symbol

    loaded = st.session_state.get("options_symbol")
    if not loaded:
        st.info("Load a ticker to scan listed options. Massive is preferred when configured; Yahoo is used as fallback.")
        return

    budget = float(st.session_state.get("options_budget", budget))
    hold_days = int(st.session_state.get("options_hold_days", hold_days))
    history = cached_options_history(loaded)
    spot = latest_close(history)
    expirations, expiration_source = cached_expirations(loaded, settings)
    snapshot = volatility_snapshot(history)
    fallback_vol = snapshot.forecast_vol or snapshot.historical_vol_21d or 0.20
    vol_forecasts = cached_volatility_forecast_table(loaded, hold_days)
    forecast_vol = select_forecast_vol(
        vol_forecasts,
        horizon_days=hold_days,
        preferred_model="ensemble",
        fallback=fallback_vol,
    ) or fallback_vol
    if spot <= 0:
        st.warning("Could not resolve a usable market price for this ticker.")
        return

    metrics = st.columns(6)
    metrics[0].metric("Ticker", loaded)
    metrics[1].metric("Spot", money(spot))
    metrics[2].metric("Ensemble Vol", pct(forecast_vol))
    metrics[3].metric("RSI 14", f"{snapshot.rsi_14:.1f}")
    metrics[4].metric("Expirations", str(len(expirations)))
    metrics[5].metric("Data Source", expiration_source)
    with st.expander("Volatility Forecast Models", expanded=False):
        st.caption(
            "Forecasts are persisted to "
            f"`{display_path(DEFAULT_VOL_FORECAST_DB_PATH)}`. "
            "The scanner uses the Ensemble row nearest your target hold window."
        )
        render_dark_table(format_vol_forecast_display(vol_forecasts, hold_days=hold_days), max_height_px=360)

    reference_expiry = choose_expiration(expirations, hold_days) if expirations else ""
    if reference_expiry:
        reference_calls, reference_puts, reference_chain_source = cached_option_chain(loaded, reference_expiry, settings)
    else:
        reference_calls, reference_puts, reference_chain_source = pd.DataFrame(), pd.DataFrame(), "n/a"
    price_targets = cached_price_targets(loaded, settings.fmp_api_key)
    sentiment_payload = cached_directional_sentiment(loaded, settings.fmp_api_key)
    directional_lens = build_directional_lens(
        loaded,
        history,
        spot=spot,
        price_targets=price_targets,
        sentiment_payload=sentiment_payload,
        calls=reference_calls,
        puts=reference_puts,
    )

    fit_tab, direction_tab, scanner_tab, lab_tab = st.tabs(ops_tabs(OPS_LANG, "strategy_fit_tabs"))
    with fit_tab:
        render_strategy_fit(
            loaded,
            history,
            expirations,
            spot,
            forecast_vol,
            hold_days,
            budget,
            settings,
            directional_lens=directional_lens,
        )
    with direction_tab:
        render_directional_lens(directional_lens, chain_source=reference_chain_source)
    with scanner_tab:
        render_scanner(loaded, spot, expirations, budget, hold_days, forecast_vol, settings)
    with lab_tab:
        render_payoff_lab(spot, forecast_vol, hold_days)


def render_opportunity_decision_pad(
    *,
    symbol: str,
    action_bucket: str,
    action_reason: str,
    directional_lens: DirectionalLensResult,
    news_nlp: NewsNLPResult,
    route_frame: pd.DataFrame,
    target_upside: float | None,
    expiration_count: int,
    market_iv: float,
    forecast_vol: float,
) -> None:
    st.subheader("Thesis Builder")
    st.caption("Turn the loaded ticker bundle into a human trade note. This does not route or approve orders.")
    rows = load_scratchpad()
    direction_label = str(directional_lens.summary.get("primary_direction") or "Neutral")
    direction_score = safe_num(directional_lens.summary.get("primary_score"), 0.0)
    news_label = str(news_nlp.summary.get("sentiment_label") or "No data")
    news_score = safe_num(news_nlp.summary.get("sentiment_score"), 0.0)
    route_label = primary_route(route_frame)
    catalyst_board = build_news_catalyst_board(news_nlp.article_frame)
    default_thesis = build_thesis_draft(
        symbol=symbol,
        action_bucket=action_bucket,
        action_reason=action_reason,
        route_frame=route_frame,
        direction=direction_label,
        direction_score=direction_score,
        news_label=news_label,
        news_score=news_score,
        top_topics=str(news_nlp.summary.get("top_topics") or ""),
        top_keywords=str(news_nlp.summary.get("top_keywords") or ""),
        catalyst_evidence=catalyst_evidence_text(catalyst_board),
    )
    checklist = build_decision_checklist_frame(
        route_frame=route_frame,
        article_count=int(news_nlp.summary.get("article_count") or 0),
        target_upside=target_upside,
        expiration_count=expiration_count,
        market_iv=market_iv,
        forecast_vol=forecast_vol,
    )

    metrics = st.columns(5)
    metrics[0].metric("Primary Route", route_label)
    metrics[1].metric("Action Bucket", action_bucket)
    metrics[2].metric("Direction", direction_label)
    metrics[3].metric("News Tone", news_label)
    metrics[4].metric("Vol Edge", pct(forecast_vol - market_iv))

    builder_tab, saved_tab = st.tabs(ops_tabs(OPS_LANG, "builder_tabs"))
    with builder_tab:
        render_dark_table(format_decision_checklist_display(checklist), max_height_px=260)
        vehicle_options = ["shares", "long options", "defined-risk spread", "income / short vol", "watch / wait", "hedge", "none yet"]
        route_vehicle = route_label.lower()
        vehicle_index = next(
            (index for index, value in enumerate(vehicle_options) if value in route_vehicle or route_vehicle in value),
            4 if route_label == "Watch / wait" else 6,
        )
        with st.form(f"decision_pad_{symbol}"):
            cols = st.columns(5)
            direction = cols[0].selectbox("Direction", ["watch", "long", "short", "hedge", "income", "avoid"], key=f"decision_direction_{symbol}")
            horizon = cols[1].selectbox("Horizon", ["intraday", "days", "weeks", "months", "long-term"], index=2, key=f"decision_horizon_{symbol}")
            status = cols[2].selectbox("Status", ["idea", "watching", "paper candidate", "entered", "closed", "rejected"], key=f"decision_status_{symbol}")
            vehicle = cols[3].selectbox("Vehicle", vehicle_options, index=vehicle_index, key=f"decision_vehicle_{symbol}")
            conviction = cols[4].selectbox("Conviction", ["low", "medium", "high"], index=1, key=f"decision_conviction_{symbol}")
            trigger = st.text_input("Trigger", value="Wait for price/volume confirmation before acting.")
            invalidation = st.text_input("Invalidation", value="Reassess if the directional/news/vol read flips against the route.")
            risk_plan = st.text_input("Risk Plan", value="Define max loss before any proposal; avoid sizing from conviction alone.")
            thesis = st.text_area("Thesis Draft", value=default_thesis, height=260)
            submitted = st.form_submit_button(T("save_idea"))
        if submitted:
            rows.insert(
                0,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "symbol": symbol,
                    "direction": direction,
                    "horizon": horizon,
                    "status": status,
                    "vehicle": vehicle,
                    "conviction": conviction,
                    "primary_route": route_label,
                    "action_bucket": action_bucket,
                    "action_reason": action_reason,
                    "direction_read": direction_label,
                    "direction_score": direction_score,
                    "news_tone": news_label,
                    "news_score": news_score,
                    "trigger": trigger,
                    "invalidation": invalidation,
                    "risk_plan": risk_plan,
                    "thesis": thesis,
                },
            )
            save_scratchpad(rows[:250])
            st.success(f"Saved idea for {symbol}.")
    symbol_rows = [row for row in rows if str(row.get("symbol", "")).upper() == symbol.upper()]
    with saved_tab:
        if symbol_rows:
            st.caption(f"Saved notes for {symbol}: `{display_path(SCRATCHPAD_PATH)}`")
            render_dark_table(format_saved_ideas_display(symbol_rows[:25]), max_height_px=360)
            with st.expander("Latest Full Thesis", expanded=False):
                st.write(str(symbol_rows[0].get("thesis") or ""))
        else:
            st.info("No saved ideas for this ticker yet.")


def render_opportunity_hub(settings: Any) -> None:
    st.subheader("Opportunity Hub")
    controls = st.columns([1.3, 0.9, 0.9, 0.8])
    symbol = controls[0].text_input("Ticker", value=st.session_state.get("workbench_symbol", "AAPL")).upper().strip()
    budget = controls[1].number_input("Options Budget / Max Risk", min_value=100.0, value=float(st.session_state.get("hub_budget", 1000.0)), step=100.0)
    hold_days = controls[2].number_input("Target Hold Days", min_value=1, max_value=365, value=int(st.session_state.get("hub_hold_days", 30)), step=1)
    if controls[3].button("Load Hub", type="primary", disabled=not symbol, use_container_width=True):
        st.session_state["hub_symbol"] = symbol
        st.session_state["workbench_symbol"] = symbol
        st.session_state["hub_budget"] = float(budget)
        st.session_state["hub_hold_days"] = int(hold_days)

    loaded = st.session_state.get("hub_symbol")
    if not loaded:
        st.info("Choose a ticker and click Load Hub to load the combined valuation, options, and news cockpit.")
        return

    budget = float(st.session_state.get("hub_budget", budget))
    hold_days = int(st.session_state.get("hub_hold_days", hold_days))
    progress, status = progress_bar(
        f"Loading {loaded} opportunity hub",
        estimate_seconds=35,
    )
    progress_step(progress, status, 5, "Loading fundamentals and valuation snapshot...")
    data, _financials, _historicals, mandates = cached_fundamental_data(loaded, settings.fmp_api_key)
    progress_step(progress, status, 18, "Loading local market history cache...")
    history = cached_options_history(loaded)
    progress_step(progress, status, 30, "Loading option expirations...")
    expirations, expiration_source = cached_expirations(loaded, settings)
    progress_step(progress, status, 42, "Loading analyst targets...")
    price_targets = cached_price_targets(loaded, settings.fmp_api_key)
    progress_step(progress, status, 52, "Loading news/article cache...")
    articles = cached_news_articles(loaded, settings.fmp_api_key)
    progress_step(progress, status, 60, "Scoring news tone, topics, and catalysts...")
    news_nlp = cached_news_nlp(loaded, news_articles_fingerprint(articles), articles)

    spot = latest_close(history) or safe_num(data.get("price"), 0.0)
    if spot <= 0:
        finish_progress(progress, status)
        st.warning("Could not resolve a usable market price for this ticker.")
        return

    snapshot = volatility_snapshot(history)
    fallback_vol = snapshot.forecast_vol or snapshot.historical_vol_21d or 0.20
    progress_step(progress, status, 70, "Loading volatility model forecasts...")
    vol_forecasts = cached_volatility_forecast_table(loaded, hold_days)
    forecast_vol = select_forecast_vol(
        vol_forecasts,
        horizon_days=hold_days,
        preferred_model="ensemble",
        fallback=fallback_vol,
    ) or fallback_vol
    reference_expiry = choose_expiration(expirations, hold_days) if expirations else ""
    if reference_expiry:
        progress_step(progress, status, 80, f"Loading option chain for {reference_expiry}...")
        reference_calls, reference_puts, reference_chain_source = cached_option_chain(loaded, reference_expiry, settings)
    else:
        reference_calls, reference_puts, reference_chain_source = pd.DataFrame(), pd.DataFrame(), "n/a"
    market_iv = estimate_atm_market_iv(reference_calls, reference_puts, spot, forecast_vol)
    progress_step(progress, status, 88, "Building directional lens...")
    directional_lens = build_directional_lens(
        loaded,
        history,
        spot=spot,
        price_targets=price_targets,
        sentiment_payload={"news": news_nlp.article_frame.to_dict("records")},
        calls=reference_calls,
        puts=reference_puts,
    )
    progress_step(progress, status, 94, "Checking provider/cache status...")
    news_status = news_cache_status(loaded)
    provider_frame = nlp_provider_status(
        fmp_key=settings.fmp_api_key,
        openai_key=getattr(settings, "openai_api_key", None),
        x_bearer_token=os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
    )
    progress_step(progress, status, 100, f"{loaded} opportunity hub ready.")
    finish_progress(progress, status)
    target = safe_num(price_targets.get("targetConsensus"), 0.0)
    target_upside = (target / spot) - 1 if spot > 0 and target > 0 else 0.0
    direction_label = str(directional_lens.summary.get("primary_direction") or "Neutral")
    direction_score = safe_num(directional_lens.summary.get("primary_score"), 0.0)
    news_label = str(news_nlp.summary.get("sentiment_label") or "No data")
    news_score = safe_num(news_nlp.summary.get("sentiment_score"), 0.0)
    action_bucket, action_reason = opportunity_action_bucket(
        spot=spot,
        price_targets=price_targets,
        forecast_vol=forecast_vol,
        market_iv=market_iv,
        directional_lens=directional_lens,
        news_nlp=news_nlp,
    )

    command = st.columns(6)
    command[0].metric("Ticker", loaded)
    command[1].metric("Spot", money(spot))
    command[2].metric("Action Bucket", action_bucket)
    command[3].metric("Direction", direction_label)
    command[4].metric("VRP", pct(market_iv - forecast_vol))
    command[5].metric("News Tone", news_label)
    st.info(action_reason)

    lens_frame, route_frame = render_opportunity_decision_cockpit(
        spot=spot,
        target_consensus=target,
        direction=direction_label,
        direction_score=direction_score,
        news_label=news_label,
        news_score=news_score,
        forecast_vol=forecast_vol,
        market_iv=market_iv,
        rsi_14=snapshot.rsi_14,
        expiration_count=len(expirations),
    )

    overview_tab, valuation_tab, options_tab, news_tab, ai_tab, decision_tab = st.tabs(
        ops_tabs(OPS_LANG, "workbench_tabs")
    )
    with overview_tab:
        overview_cols = st.columns(4)
        overview_cols[0].metric("Analyst Upside", pct(target_upside) if target else "missing")
        overview_cols[1].metric("Ensemble Vol", pct(forecast_vol))
        overview_cols[2].metric("ATM IV", pct(market_iv))
        overview_cols[3].metric("Reference Expiry", reference_expiry or "n/a")
        render_dark_table(format_opportunity_lens_display(lens_frame), max_height_px=260)
        render_dark_table(format_vehicle_route_display(route_frame), max_height_px=260)
        if data:
            render_stock_snapshot(loaded, data)
        else:
            st.warning("Fundamental valuation payload is unavailable. Options/news sections can still be reviewed.")
        render_directional_lens(directional_lens, chain_source=reference_chain_source)
    with valuation_tab:
        if not data:
            st.info("Valuation data unavailable.")
        else:
            value_tabs = st.tabs(ops_tabs(OPS_LANG, "valuation_tabs"))
            with value_tabs[0]:
                render_dcf(data, loaded, settings.fmp_api_key, settings.rapid_api_key)
            with value_tabs[1]:
                render_price_targets(loaded, settings.fmp_api_key)
            with value_tabs[2]:
                render_peers(loaded, settings.fmp_api_key)
            with value_tabs[3]:
                render_supply_chain_tab(loaded, data, settings)
            with value_tabs[4]:
                if not mandates:
                    st.info("Checklist data unavailable.")
                else:
                    score = sum(1 for passed in mandates.values() if passed)
                    st.progress(score / max(len(mandates), 1), text=f"{score}/{len(mandates)} checks passed")
                    render_dark_table(
                        pd.DataFrame({"Criterion": list(mandates), "Passed": list(mandates.values())}),
                    )
    with options_tab:
        option_tabs = st.tabs(ops_tabs(OPS_LANG, "workbench_option_tabs"))
        with option_tabs[0]:
            render_options_playbook(
                symbol=loaded,
                calls=reference_calls,
                puts=reference_puts,
                spot=spot,
                expiry=reference_expiry,
                expirations=expirations,
                budget=budget,
                hold_days=hold_days,
                forecast_vol=forecast_vol,
                market_iv=market_iv,
                chain_source=reference_chain_source,
                settings=settings,
                route_frame=route_frame,
            )
        with option_tabs[1]:
            render_strategy_fit(
                loaded,
                history,
                expirations,
                spot,
                forecast_vol,
                hold_days,
                budget,
                settings,
                directional_lens=directional_lens,
            )
        with option_tabs[2]:
            render_scanner(loaded, spot, expirations, budget, hold_days, forecast_vol, settings)
        with option_tabs[3]:
            render_payoff_lab(spot, forecast_vol, hold_days)
        with option_tabs[4]:
            st.caption(
                "Forecasts are persisted to "
                f"`{display_path(DEFAULT_VOL_FORECAST_DB_PATH)}`. "
                "The scanner uses the Ensemble row nearest your target hold window."
            )
            render_dark_table(format_vol_forecast_display(vol_forecasts, hold_days=hold_days), max_height_px=360)
    with news_tab:
        st.caption(f"News/NLP cache path: `{display_path(DEFAULT_NEWS_NLP_DB_PATH)}`")
        render_news_nlp_section(news_nlp, cache_state=news_status, provider_frame=provider_frame)
    with ai_tab:
        render_ai_outlook_section(
            symbol=loaded,
            data=data,
            articles=articles,
            settings=settings,
        )
    with decision_tab:
        render_opportunity_decision_pad(
            symbol=loaded,
            action_bucket=action_bucket,
            action_reason=action_reason,
            directional_lens=directional_lens,
            news_nlp=news_nlp,
            route_frame=route_frame,
            target_upside=target_upside if target else None,
            expiration_count=len(expirations),
            market_iv=market_iv,
            forecast_vol=forecast_vol,
        )


def render_option_payoff_surface(
    *,
    spot: float,
    strike: float,
    premium: float,
    option_type: str,
    side: str,
    volatility: float,
    expiry_days: int,
) -> None:
    price_grid = np.linspace(max(0.01, spot * 0.65), spot * 1.35, 60)
    elapsed_grid = np.linspace(0, max(expiry_days, 1), 36)
    surface = []
    for elapsed in elapsed_grid:
        remaining_years = max(expiry_days - elapsed, 0) / 365.0
        row = []
        for price in price_grid:
            option_value = black_scholes_price(
                spot=float(price),
                strike=float(strike),
                time_to_expiry=remaining_years,
                rate=0.045,
                volatility=max(volatility, 0.001),
                option_type=option_type,
            )
            pnl = option_value - premium if side == "long" else premium - option_value
            row.append(pnl)
        surface.append(row)

    fig = go.Figure(
        data=[
            go.Surface(
                x=price_grid,
                y=elapsed_grid,
                z=np.array(surface),
                colorscale="RdYlGn",
                colorbar=dict(title="P&L/share"),
                contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "#111827"}},
            )
        ]
    )
    fig.update_layout(
        height=520,
        margin=dict(t=20, b=20, l=10, r=10),
        scene=dict(
            xaxis_title="Underlying Price",
            yaxis_title="Days Elapsed",
            zaxis_title="P&L / Share",
            camera=dict(eye=dict(x=1.4, y=1.5, z=0.9)),
        ),
    )
    style_dark_plotly(fig, hovermode=None)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_payoff_lab(spot: float, forecast_vol: float, hold_days: int) -> None:
    cols = st.columns(6)
    option_type = cols[0].selectbox("Option", ["call", "put"])
    side = cols[1].selectbox("Side", ["long", "short"])
    strike = cols[2].number_input("Strike", min_value=0.01, value=float(round(spot, 2)), step=1.0)
    premium = cols[3].number_input("Premium", min_value=0.0, value=2.50, step=0.05)
    expiry_days = cols[4].number_input("Expiry Days", min_value=1, max_value=1095, value=max(int(hold_days), 30), step=1)
    simulations = cols[5].number_input("Simulations", min_value=500, max_value=20000, value=5000, step=500)
    render_option_payoff_surface(
        spot=spot,
        strike=float(strike),
        premium=float(premium),
        option_type=option_type,
        side=side,
        volatility=forecast_vol,
        expiry_days=int(expiry_days),
    )
    sim = simulate_single_option(
        spot=spot,
        strike=strike,
        premium=premium,
        option_type=option_type,
        side=side,
        days_to_hold=min(hold_days, int(expiry_days)),
        volatility=forecast_vol,
        simulations=int(simulations),
    )
    metrics = st.columns(5)
    metrics[0].metric("PoP", pct(sim.probability_of_profit))
    metrics[1].metric("EV / Share", signed_money(sim.expected_value))
    metrics[2].metric("VaR 95 / Share", signed_money(sim.value_at_risk_95))
    metrics[3].metric("Worst / Share", signed_money(sim.worst_case))
    metrics[4].metric("Best / Share", signed_money(sim.best_case))


def render_scratchpad_tab() -> None:
    st.subheader("Trade Scratchpad")
    rows = load_scratchpad()
    with st.form("scratchpad_form"):
        cols = st.columns(4)
        symbol = cols[0].text_input("Symbol", value=st.session_state.get("workbench_symbol", "")).upper().strip()
        direction = cols[1].selectbox("Direction", ["watch", "long", "short", "hedge", "income", "avoid"])
        horizon = cols[2].selectbox("Horizon", ["intraday", "days", "weeks", "months", "long-term"])
        status = cols[3].selectbox("Status", ["idea", "watching", "paper candidate", "entered", "closed", "rejected"])
        thesis = st.text_area("Thesis / trigger / invalidation")
        submitted = st.form_submit_button(T("save_idea"), disabled=not symbol)
    if submitted:
        rows.insert(
            0,
            {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbol": symbol,
                "direction": direction,
                "horizon": horizon,
                "status": status,
                "thesis": thesis,
            },
        )
        save_scratchpad(rows[:250])
        st.success(f"Saved idea for {symbol}.")
    if rows:
        st.caption(f"Scratchpad path: `{display_path(SCRATCHPAD_PATH)}`")
        render_dark_table(pd.DataFrame(rows), max_height_px=460)
    else:
        st.info("No discretionary trade ideas saved yet.")


def render_api_status_tab(settings: Any) -> None:
    st.subheader("API Status")
    rows = [
        {"Service": "FMP", "Env Key": "FMP_API_KEY", "Status": present(settings.fmp_api_key), "Used By": "valuation, peers, analyst targets, news/articles"},
        {
            "Service": "Massive",
            "Env Key": "MASSIVE_API_KEY",
            "Status": present(settings.massive_api_key or settings.options_api_key or settings.polygon_api_key),
            "Used By": "options and market data",
        },
        {"Service": "RapidAPI", "Env Key": "RAPID_API_KEY", "Status": present(settings.rapid_api_key), "Used By": "supplemental discretionary data"},
        {
            "Service": "LLM Evidence",
            "Env Key": "ZAI_API_KEY / OPENAI_API_KEY",
            "Status": present(getattr(settings, "llm_api_key", None)),
            "Used By": f"{getattr(settings, 'llm_provider', 'missing')} / {getattr(settings, 'llm_model', 'missing')}",
        },
        {"Service": "News/NLP Cache", "Env Key": "n/a", "Status": "local", "Used By": display_path(DEFAULT_NEWS_NLP_DB_PATH)},
    ]
    render_dark_table(pd.DataFrame(rows))
    st.subheader("Sentiment Provider Readiness")
    provider_frame = nlp_provider_status(
        fmp_key=settings.fmp_api_key,
        openai_key=getattr(settings, "openai_api_key", None),
        x_bearer_token=os.getenv("X_BEARER_TOKEN") or os.getenv("TWITTER_BEARER_TOKEN"),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
    )
    render_dark_table(format_provider_status_display(provider_frame), max_height_px=360)
    st.info("Keys are loaded through `.env`, server env, or legacy fallback files. No key values are displayed here.")


settings = load_settings()

page_header(
    title="Discretionary Workbench",
    title_zh="主观交易工作台",
    subtitle="Daily discretionary cockpit: watchlist, combined valuation/options/news hub, and API readiness.",
    subtitle_zh="日常主观决策驾驶舱：自选股、估值/期权/新闻一体化机会中心与 API 状态。",
    language=OPS_LANG,
)

active_symbol = st.session_state.get("workbench_symbol", "AAPL")
top = st.columns(5)
top[0].metric(T("active_symbol", "Active Symbol"), active_symbol)
top[1].metric("FMP", present(settings.fmp_api_key))
top[2].metric("Massive", present(settings.massive_api_key or settings.polygon_api_key or settings.options_api_key))
top[3].metric(T("watchlist", "Watchlist"), str(len(load_stock_watchlist())))
top[4].metric(T("saved_ideas", "Saved Ideas"), str(len(load_scratchpad())))

if st.session_state.get("workbench_view") not in WORKBENCH_VIEWS:
    st.session_state["workbench_view"] = "Watchlist"

active_view = st.session_state.get("workbench_view", "Watchlist")
render_workbench_nav(active_view)

if active_view == "Watchlist":
    render_watchlist_tab()
elif active_view == "Opportunity Hub":
    render_opportunity_hub(settings)
else:
    render_api_status_tab(settings)
