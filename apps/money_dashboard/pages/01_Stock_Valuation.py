"""Native stock valuation page for the money dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.config import load_settings  # noqa: E402
from oqp.investing import (  # noqa: E402
    DEFAULT_STOCK_WATCHLIST_PATH,
    RATIO_CATEGORIES,
    VALUATION_MULTIPLE_COLUMNS,
    add_stock_watchlist_symbol,
    calculate_dcf_valuation,
    fetch_fundamental_data,
    fetch_peer_comparison,
    fetch_price_target_consensus,
    format_compact_currency,
    load_stock_watchlist,
    remove_stock_watchlist_symbol,
    safe_num,
)
from oqp.portfolio import (  # noqa: E402
    DEFAULT_IBKR_METRICS_PATH,
    default_portfolio_ledger_path,
    load_latest_live_positions,
)


try:
    from google import genai
    from google.genai import types

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


st.set_page_config(page_title="Stock Valuation", layout="wide", page_icon="$")


TEXT = {
    "English": {
        "title": "Stock Valuation",
        "subtitle": "Fundamentals, DCF, peer multiples, analyst targets, and live portfolio context.",
        "language": "Language",
        "credentials": "Credentials",
        "workspace": "Workspace",
        "ticker": "Target ticker",
        "load": "Load valuation",
        "save": "Save ticker",
        "remove": "Remove ticker",
        "watchlist": "Watchlist",
        "new_search": "-- New Search --",
        "valuation_model": "Valuation Model",
        "standard": "Standard FCF Growth",
        "margin": "Revenue & Margin Expansion",
        "wacc": "Discount Rate (WACC) %",
        "terminal_growth": "Terminal Growth Rate %",
        "fcf_g1": "FCF Growth Y1-5 %",
        "fcf_g2": "FCF Growth Y6-10 %",
        "rev_g1": "Revenue Growth Y1-5 %",
        "rev_g2": "Revenue Growth Y6-10 %",
        "target_margin": "Target FCF Margin Y10 %",
        "load_prompt": "Enter a ticker, then load the valuation workspace from the sidebar.",
    },
    "Chinese": {
        "title": "股票估值",
        "subtitle": "基本面、DCF、同行估值、分析师目标价，以及实时持仓上下文。",
        "language": "语言",
        "credentials": "凭证",
        "workspace": "工作区",
        "ticker": "目标代码",
        "load": "加载估值",
        "save": "保存代码",
        "remove": "移除代码",
        "watchlist": "自选股",
        "new_search": "-- 新搜索 --",
        "valuation_model": "估值模型",
        "standard": "标准自由现金流增长",
        "margin": "收入与利润率扩张",
        "wacc": "折现率 (WACC) %",
        "terminal_growth": "永续增长率 %",
        "fcf_g1": "自由现金流增长 1-5年 %",
        "fcf_g2": "自由现金流增长 6-10年 %",
        "rev_g1": "收入增长 1-5年 %",
        "rev_g2": "收入增长 6-10年 %",
        "target_margin": "第10年目标FCF利润率 %",
        "load_prompt": "在侧边栏输入代码，然后加载估值工作区。",
    },
}


def st_secret(*names: str) -> str:
    for name in names:
        try:
            value = st.secrets.get(name, "")
        except Exception:
            value = ""
        if value:
            return str(value)
    return ""


def clean_secret(value: object) -> str:
    return str(value or "").strip().strip('"').strip("'")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def money(value: object, digits: int = 2) -> str:
    return f"${safe_num(value):,.{digits}f}"


def percent(value: object, digits: int = 1) -> str:
    return f"{safe_num(value) * 100:+.{digits}f}%"


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fundamental_data(symbol: str, fmp_key: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bool]]:
    return fetch_fundamental_data(symbol, fmp_key)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_price_history(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    try:
        return yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_price_targets(symbol: str, fmp_key: str) -> dict[str, Any]:
    return fetch_price_target_consensus(fmp_key, symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_peer_comparison(symbol: str, fmp_key: str):
    return fetch_peer_comparison(fmp_key, symbol)


def live_equity_positions() -> pd.DataFrame:
    positions = load_latest_live_positions(default_portfolio_ledger_path())
    if positions.empty:
        return pd.DataFrame()
    out = positions.copy()
    out["asset_type_lower"] = out["asset_type"].astype(str).str.lower()
    out = out[out["asset_type_lower"].isin(["equity", "stock", "etf"])]
    if out.empty:
        return pd.DataFrame()
    for column in ("shares", "avg_cost", "current_price", "unrealized_pnl"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["market_value"] = out["shares"] * out["current_price"]
    out["cost_basis"] = out["shares"] * out["avg_cost"]
    return out.drop(columns=["asset_type_lower"])


def portfolio_context(symbol: str, data: dict[str, Any]) -> dict[str, float]:
    positions = live_equity_positions()
    metrics = read_json(DEFAULT_IBKR_METRICS_PATH)
    if positions.empty:
        return {
            "shares": 0.0,
            "avg_cost": 0.0,
            "market_value": 0.0,
            "cash": safe_num(metrics.get("Available_Cash_USD")),
            "total_nav": safe_num(metrics.get("Total_NAV_USD")),
        }

    target = positions[positions["ticker"].astype(str).str.upper() == symbol.upper()]
    shares = safe_num(target["shares"].sum()) if not target.empty else 0.0
    cost = safe_num(target["cost_basis"].sum()) if not target.empty else 0.0
    live_price = safe_num(data.get("price"))
    avg_cost = cost / shares if shares else 0.0
    market_value = shares * live_price if live_price else safe_num(target["market_value"].sum())
    return {
        "shares": shares,
        "avg_cost": avg_cost,
        "market_value": market_value,
        "cash": safe_num(metrics.get("Available_Cash_USD")),
        "total_nav": safe_num(metrics.get("Total_NAV_USD")),
    }


def render_portfolio_holdings(symbol: str) -> None:
    positions = live_equity_positions()
    if positions.empty:
        st.info("No runtime equity holdings found yet.")
        return

    display = positions[
        [
            "date",
            "broker",
            "ticker",
            "asset_type",
            "shares",
            "avg_cost",
            "current_price",
            "market_value",
            "unrealized_pnl",
            "currency",
        ]
    ].rename(
        columns={
            "date": "Date",
            "broker": "Broker",
            "ticker": "Ticker",
            "asset_type": "Asset Type",
            "shares": "Shares",
            "avg_cost": "Avg Cost",
            "current_price": "Current Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L",
            "currency": "Currency",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    selected = positions[positions["ticker"].astype(str).str.upper() == symbol.upper()]
    if not selected.empty:
        st.caption(f"{symbol.upper()} is currently present in the runtime portfolio ledger.")


def render_snapshot(symbol: str, data: dict[str, Any]) -> None:
    st.subheader(f"Executive Snapshot: {symbol} ({data.get('sector', 'Unknown')})")
    cols = st.columns(4)
    cols[0].metric("Current Price", money(data.get("price")))
    cols[1].metric("Market Cap", format_compact_currency(safe_num(data.get("market_cap"))))
    cols[2].metric("P/E Ratio", f"{safe_num(data.get('pe')):.1f}")
    cols[3].metric("ROCE", f"{safe_num(data.get('roce')) * 100:.1f}%")


def render_quarterly_health(data: dict[str, Any]) -> None:
    q_assets = safe_num(data.get("q_curr_assets"))
    q_liabilities = safe_num(data.get("q_curr_liab"))
    q_equity = safe_num(data.get("q_equity"))
    q_revenue = safe_num(data.get("q_rev"))
    current_ratio = q_assets / q_liabilities if q_liabilities > 0 else 0.0
    quick_ratio = (q_assets - safe_num(data.get("q_inventory"))) / q_liabilities if q_liabilities > 0 else 0.0
    debt_equity = safe_num(data.get("q_total_debt")) / q_equity if q_equity > 0 else 0.0
    gross_margin = safe_num(data.get("q_gross")) / q_revenue if q_revenue > 0 else 0.0
    operating_margin = safe_num(data.get("q_op_inc")) / q_revenue if q_revenue > 0 else 0.0

    st.markdown("### Fundamental Quick Check")
    first = st.columns(3)
    first[0].metric("Current Ratio", f"{current_ratio:.2f}", "Target > 1.5", delta_color="off")
    first[1].metric("Quick Ratio", f"{quick_ratio:.2f}", "Target > 1.0", delta_color="off")
    first[2].metric("Debt/Equity", f"{debt_equity:.2f}", "Lower is safer", delta_color="inverse")
    second = st.columns(3)
    second[0].metric("Quarterly FCF", format_compact_currency(safe_num(data.get("q_fcf"))))
    second[1].metric("Gross Margin", f"{gross_margin * 100:.1f}%")
    second[2].metric("Operating Margin", f"{operating_margin * 100:.1f}%")


def render_checklist(mandates: dict[str, bool]) -> None:
    st.markdown("### 10-Point Quality Checklist")
    if not mandates:
        st.info("Checklist data unavailable.")
        return
    score = sum(1 for passed in mandates.values() if passed)
    st.progress(score / 10.0, text=f"Score: {score}/10")
    rows = [
        {"Criterion": criterion, "Status": "Pass" if passed else "Fail"}
        for criterion, passed in mandates.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_dcf(
    data: dict[str, Any],
    *,
    model: str,
    wacc: float,
    terminal_growth: float,
    fcf_growth_1: float,
    fcf_growth_2: float,
    revenue_growth_1: float,
    revenue_growth_2: float,
    target_fcf_margin: float,
) -> None:
    try:
        valuation = calculate_dcf_valuation(
            data,
            model=model,
            wacc=wacc,
            terminal_growth=terminal_growth,
            fcf_growth_1=fcf_growth_1,
            fcf_growth_2=fcf_growth_2,
            revenue_growth_1=revenue_growth_1,
            revenue_growth_2=revenue_growth_2,
            target_fcf_margin=target_fcf_margin,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown("### DCF Valuation")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Intrinsic Value", money(valuation.fair_value_per_share))
    metric_cols[1].metric("Margin of Safety", percent(valuation.margin_of_safety))

    bridge = pd.DataFrame(
        [
            {"Valuation Metric": "PV of 10-Yr Cash Flows", "Amount": sum(valuation.present_value_fcf)},
            {"Valuation Metric": "PV of Terminal Value", "Amount": valuation.present_value_terminal},
            {"Valuation Metric": "Enterprise Value", "Amount": valuation.enterprise_value},
            {"Valuation Metric": "+ Total Cash", "Amount": safe_num(data.get("total_cash"))},
            {"Valuation Metric": "- Total Debt", "Amount": -safe_num(data.get("total_debt"))},
            {"Valuation Metric": "Equity Value", "Amount": valuation.equity_value},
        ]
    )
    bridge["Amount"] = bridge["Amount"].map(lambda value: format_compact_currency(float(value)))
    st.dataframe(bridge, use_container_width=True, hide_index=True)

    fig = go.Figure()
    years = [f"Year {index}" for index in range(1, 11)]
    fig.add_bar(x=years, y=valuation.future_fcf, name="Projected FCF", marker_color="#2563EB")
    fig.add_scatter(x=years, y=valuation.present_value_fcf, name="Discounted PV", mode="lines+markers")
    fig.update_layout(height=320, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sensitivity Matrix")
    w_range = [wacc - 0.01, wacc, wacc + 0.01]
    cf_range = [0.9, 1.0, 1.1]
    rows = []
    for cf_multiplier in cf_range:
        row = {}
        for w_adj in w_range:
            try:
                adjusted = calculate_dcf_valuation(
                    {
                        **data,
                        "fcf_ttm": safe_num(data.get("fcf_ttm")) * cf_multiplier,
                        "ttm_revenue": safe_num(data.get("ttm_revenue")) * cf_multiplier,
                    },
                    model=model,
                    wacc=w_adj,
                    terminal_growth=terminal_growth,
                    fcf_growth_1=fcf_growth_1,
                    fcf_growth_2=fcf_growth_2,
                    revenue_growth_1=revenue_growth_1,
                    revenue_growth_2=revenue_growth_2,
                    target_fcf_margin=target_fcf_margin,
                )
                row[f"{w_adj * 100:.1f}% WACC"] = adjusted.fair_value_per_share
            except ValueError:
                row[f"{w_adj * 100:.1f}% WACC"] = None
        rows.append(row)
    matrix = pd.DataFrame(rows, index=["Pessimistic (-10% CF)", "Base Case", "Optimistic (+10% CF)"])
    st.dataframe(matrix.style.format("${:.2f}", na_rep="n/a").background_gradient(cmap="viridis", axis=None), use_container_width=True)


def render_technicals(data: dict[str, Any]) -> None:
    hist_df = data.get("hist_df")
    if not isinstance(hist_df, pd.DataFrame) or hist_df.empty:
        st.warning("Historical price data not available for technical analysis.")
        return

    fig_price = go.Figure()
    fig_price.add_scatter(x=hist_df.index, y=hist_df["BB_Upper"], name="Upper BB", line=dict(width=1), showlegend=False)
    fig_price.add_scatter(
        x=hist_df.index,
        y=hist_df["BB_Lower"],
        name="Lower BB",
        fill="tonexty",
        fillcolor="rgba(37,99,235,0.08)",
        line=dict(width=1),
        showlegend=False,
    )
    fig_price.add_scatter(x=hist_df.index, y=hist_df["SMA_50"], name="50-Day SMA", line=dict(width=1.5, dash="dash"))
    fig_price.add_scatter(x=hist_df.index, y=hist_df["SMA_200"], name="200-Day SMA", line=dict(width=1.5, dash="dash"))
    fig_price.add_scatter(x=hist_df.index, y=hist_df["Close"], name="Price", line=dict(width=2))
    fig_price.update_layout(height=420, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_price, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_scatter(x=hist_df.index, y=hist_df["MACD"], name="MACD")
    fig_macd.add_scatter(x=hist_df.index, y=hist_df["Signal"], name="Signal", line=dict(dash="dot"))
    colors = ["#16A34A" if value >= 0 else "#DC2626" for value in hist_df["MACD_Hist"]]
    fig_macd.add_bar(x=hist_df.index, y=hist_df["MACD_Hist"], name="Histogram", marker_color=colors)
    fig_macd.update_layout(height=260, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig_macd, use_container_width=True)


def render_price_targets(symbol: str, fmp_key: str) -> None:
    hist_df = cached_price_history(symbol, "2y")
    targets = cached_price_targets(symbol, fmp_key)
    fig = go.Figure()
    if not hist_df.empty and "Close" in hist_df:
        fig.add_scatter(x=hist_df.index, y=hist_df["Close"], mode="lines", name="Close Price")

    target_consensus = safe_num(targets.get("targetConsensus"))
    target_high = safe_num(targets.get("targetHigh"))
    target_low = safe_num(targets.get("targetLow"))
    if target_consensus:
        fig.add_hline(y=target_consensus, line_dash="dash", line_color="#16A34A", annotation_text=f"Consensus: ${target_consensus:.2f}")
    if target_high:
        fig.add_hline(y=target_high, line_dash="dot", line_color="#D97706", annotation_text=f"High: ${target_high:.2f}")
    if target_low:
        fig.add_hline(y=target_low, line_dash="dot", line_color="#DC2626", annotation_text=f"Low: ${target_low:.2f}")

    if not targets:
        st.warning("Consensus target data not found. Check FMP key/access for this endpoint.")
    fig.update_layout(height=480, margin=dict(t=20, b=20, l=10, r=10), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_peer_comparison(symbol: str, fmp_key: str) -> None:
    peer_data = cached_peer_comparison(symbol, fmp_key)
    if peer_data.error:
        st.warning(peer_data.error)
        return

    sub_tab_1, sub_tab_2 = st.tabs(["Valuation Multiples", "Financial Ratios"])
    with sub_tab_1:
        if peer_data.metrics.empty:
            st.warning("Could not fetch peer valuation metrics.")
        else:
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
                st.dataframe(peer_data.metrics[cols], use_container_width=True, column_config=config)
            else:
                st.info("Standard valuation metrics are unavailable for this peer group.")
    with sub_tab_2:
        if peer_data.ratios.empty:
            st.warning("Could not fetch peer financial ratios.")
            return
        category = st.selectbox("Ratio category", list(RATIO_CATEGORIES.keys()))
        selected = RATIO_CATEGORIES[category]
        cols = [column for column in selected if column in peer_data.ratios.columns]
        if cols:
            config = {
                column: st.column_config.NumberColumn(
                    label=selected[column][0],
                    help=selected[column][1],
                    format="%.3f",
                )
                for column in cols
            }
            st.dataframe(peer_data.ratios[cols], use_container_width=True, column_config=config)
        else:
            st.info(f"Standard {category} ratios are unavailable for this peer group.")


def render_ai_analyst(symbol: str, data: dict[str, Any], gemini_key: str) -> None:
    st.markdown("### AI Analyst")
    if not HAS_GEMINI:
        st.error("google-genai is not installed in this environment.")
        return
    if not gemini_key:
        st.warning("Enter a Gemini API key in the sidebar to activate the AI analyst.")
        return

    context = portfolio_context(symbol, data)
    agent_mode = st.radio("Agent", ["Portfolio Planner", "Stock Researcher"], horizontal=True)
    chat_id = f"stock_valuation_{agent_mode}_{symbol}"
    if chat_id not in st.session_state:
        if agent_mode == "Portfolio Planner":
            st.session_state[chat_id] = [
                {
                    "role": "assistant",
                    "content": (
                        f"I see {context['shares']:.2f} shares of {symbol} at an average cost "
                        f"of {money(context['avg_cost'])}, with IBKR cash around {money(context['cash'])}."
                    ),
                }
            ]
        else:
            st.session_state[chat_id] = [
                {
                    "role": "assistant",
                    "content": f"I have loaded the core valuation data for {symbol}. Ask for a thesis, risk check, or catalyst map.",
                }
            ]

    for message in st.session_state[chat_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input(f"Talk to the {agent_mode}...")
    if not user_prompt:
        return

    st.session_state[chat_id].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    system_instruction = (
        "You are an institutional equity analyst. Be concise, specific, and practical. "
        "Give scenarios, risk controls, and concrete price or allocation levels when appropriate."
    )
    live_context = f"""
Target: {symbol}
Current price: {money(data.get("price"))}
Market cap: {format_compact_currency(safe_num(data.get("market_cap")))}
P/E: {safe_num(data.get("pe")):.2f}
ROCE: {safe_num(data.get("roce")) * 100:.1f}%
Quarterly FCF: {format_compact_currency(safe_num(data.get("q_fcf")))}
Shares currently held: {context["shares"]:.4f}
Average cost: {money(context["avg_cost"])}
Position market value: {money(context["market_value"])}
Available IBKR cash: {money(context["cash"])}
Total IBKR NAV: {money(context["total_nav"])}
User query: {user_prompt}
"""
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = genai.Client(api_key=gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=live_context,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.35,
                    ),
                )
                reply = response.text
            except Exception as exc:
                st.error(f"Gemini API Error: {exc}")
                return
            st.markdown(reply)
            st.session_state[chat_id].append({"role": "assistant", "content": reply})


settings = load_settings()
language_label = st.sidebar.radio("Language / 语言", ["English", "Chinese"], horizontal=True)
t = TEXT[language_label]

st.title(t["title"])
st.caption(t["subtitle"])

default_fmp = st_secret("FMP_KEY", "FMP_API_KEY") or (settings.fmp_api_key or "")
default_gemini = st_secret("GEMINI_KEY", "GEMINI_API_KEY") or (settings.gemini_api_key or "")

with st.sidebar:
    st.header(t["workspace"])
    watchlist = load_stock_watchlist()
    selected_watch = t["new_search"]
    if watchlist:
        selected_watch = st.selectbox(t["watchlist"], [t["new_search"], *watchlist])
    initial_ticker = selected_watch if selected_watch != t["new_search"] else "AAPL"
    target_ticker = st.text_input(t["ticker"], value=initial_ticker).upper().strip()

    col_save, col_load = st.columns(2)
    with col_save:
        if target_ticker and target_ticker in watchlist:
            if st.button(t["remove"], use_container_width=True):
                remove_stock_watchlist_symbol(target_ticker)
                st.rerun()
        elif target_ticker:
            if st.button(t["save"], use_container_width=True):
                add_stock_watchlist_symbol(target_ticker)
                st.rerun()
    with col_load:
        load_requested = st.button(t["load"], type="primary", use_container_width=True)

    st.caption(f"Watchlist: `{DEFAULT_STOCK_WATCHLIST_PATH.relative_to(REPO_ROOT)}`")
    st.divider()

    st.header(t["credentials"])
    fmp_key = clean_secret(st.text_input("FMP API Key", value=default_fmp, type="password"))
    gemini_key = clean_secret(st.text_input("Gemini API Key", value=default_gemini, type="password"))
    st.divider()

    st.header(t["valuation_model"])
    model_choice = st.radio(t["valuation_model"], [t["standard"], t["margin"]])
    wacc = st.number_input(t["wacc"], 1.0, 50.0, 10.0, 0.5, format="%.2f") / 100
    terminal_growth = st.number_input(t["terminal_growth"], -5.0, 10.0, 2.5, 0.1, format="%.2f") / 100


if load_requested and target_ticker:
    st.session_state["stock_valuation_loaded_ticker"] = target_ticker

loaded_ticker = st.session_state.get("stock_valuation_loaded_ticker")
if not loaded_ticker:
    st.info(t["load_prompt"])
    with st.expander("Runtime equity holdings", expanded=True):
        render_portfolio_holdings(target_ticker or "")
    st.stop()

with st.spinner(f"Loading {loaded_ticker} valuation data..."):
    data, financials, historicals, mandates = cached_fundamental_data(loaded_ticker, fmp_key)

if not data or not safe_num(data.get("price")):
    st.warning("Could not load a usable price/fundamental snapshot. Check ticker spelling and data access.")
    st.stop()

default_fcf_growth = safe_num(data.get("auto_fcf_cagr"), 15.0)
default_revenue_growth = safe_num(data.get("auto_rev_cagr"), 12.0)

with st.sidebar:
    st.markdown("### Growth Assumptions")
    if model_choice == t["standard"]:
        fcf_growth_1 = st.number_input(t["fcf_g1"], -50.0, 200.0, float(default_fcf_growth), 1.0, format="%.2f") / 100
        fcf_growth_2 = st.number_input(t["fcf_g2"], -50.0, 200.0, float(default_fcf_growth * 0.7), 1.0, format="%.2f") / 100
        revenue_growth_1 = default_revenue_growth / 100
        revenue_growth_2 = revenue_growth_1 * 0.7
        target_fcf_margin = 0.25
        dcf_model = "standard"
    else:
        revenue_growth_1 = st.number_input(t["rev_g1"], -50.0, 200.0, float(default_revenue_growth), 1.0, format="%.2f") / 100
        revenue_growth_2 = st.number_input(t["rev_g2"], -50.0, 200.0, float(default_revenue_growth * 0.7), 1.0, format="%.2f") / 100
        target_fcf_margin = st.number_input(t["target_margin"], 1.0, 100.0, 25.0, 1.0, format="%.2f") / 100
        fcf_growth_1 = default_fcf_growth / 100
        fcf_growth_2 = fcf_growth_1 * 0.7
        dcf_model = "margin"

render_snapshot(loaded_ticker, data)

with st.expander("Runtime equity holdings", expanded=False):
    render_portfolio_holdings(loaded_ticker)

tab_fundamentals, tab_technicals, tab_targets, tab_peers, tab_ai = st.tabs(
    [
        "Fundamentals & DCF",
        "Technicals",
        "Analyst Targets",
        "Peer Comparison",
        "AI Analyst",
    ]
)

with tab_fundamentals:
    left, right = st.columns([1, 1.25])
    with left:
        render_quarterly_health(data)
        render_checklist(mandates)
    with right:
        render_dcf(
            data,
            model=dcf_model,
            wacc=wacc,
            terminal_growth=terminal_growth,
            fcf_growth_1=fcf_growth_1,
            fcf_growth_2=fcf_growth_2,
            revenue_growth_1=revenue_growth_1,
            revenue_growth_2=revenue_growth_2,
            target_fcf_margin=target_fcf_margin,
        )

with tab_technicals:
    render_technicals(data)

with tab_targets:
    render_price_targets(loaded_ticker, fmp_key)

with tab_peers:
    render_peer_comparison(loaded_ticker, fmp_key)

with tab_ai:
    render_ai_analyst(loaded_ticker, data, gemini_key)
