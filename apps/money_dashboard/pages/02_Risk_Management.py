"""Native portfolio risk management page for the money dashboard."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    DEFAULT_IBKR_METRICS_PATH,
    compute_nav_drawdowns,
    default_portfolio_ledger_path,
    load_historical_nav,
    load_latest_live_positions,
)
from oqp.risk import (  # noqa: E402
    average_true_range,
    black_scholes_greeks,
    broker_risk_table,
    concentration_table,
    enrich_position_risk,
    hedge_diagnosis,
    inverse_hedge_plan,
    micro_future_multiplier,
    safe_float,
    summarize_portfolio_risk,
)


st.set_page_config(page_title="Risk Management", layout="wide", page_icon="!")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def money(value: object, digits: int = 2) -> str:
    return f"${safe_float(value):,.{digits}f}"


def signed_money(value: object, digits: int = 2) -> str:
    parsed = safe_float(value)
    sign = "+" if parsed > 0 else ""
    return f"{sign}${parsed:,.{digits}f}"


def pct(value: object, digits: int = 2) -> str:
    return f"{safe_float(value) * 100:.{digits}f}%"


def clean_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].astype(float)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(symbol: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    try:
        return yf.Ticker(symbol).history(period=period)
    except Exception:
        return pd.DataFrame()


def latest_price_from_history(history: pd.DataFrame) -> float:
    if history.empty or "Close" not in history.columns:
        return 0.0
    return safe_float(history["Close"].dropna().iloc[-1])


def annualized_vol_from_history(history: pd.DataFrame) -> float:
    if history.empty or "Close" not in history.columns:
        return 0.0
    returns = history["Close"].pct_change().dropna()
    return safe_float(returns.std()) * np.sqrt(252) if not returns.empty else 0.0


def render_metric_grid(summary) -> None:
    row_1 = st.columns(5)
    row_1[0].metric("Latest NAV", money(summary.latest_nav))
    row_1[1].metric("Daily P&L", signed_money(summary.latest_daily_pnl))
    row_1[2].metric("Portfolio Beta", f"{summary.portfolio_beta:.4f}")
    row_1[3].metric("Beta Exposure", money(summary.beta_adjusted_exposure))
    row_1[4].metric("Cash", money(summary.latest_cash))

    row_2 = st.columns(5)
    row_2[0].metric("Gross Exposure", money(summary.gross_exposure))
    row_2[1].metric("Net Delta", signed_money(summary.net_delta_exposure))
    row_2[2].metric("Max Drawdown", signed_money(summary.max_drawdown), pct(summary.max_drawdown_pct))
    row_2[3].metric("1D VaR 95", money(summary.one_day_var_95))
    row_2[4].metric("Top Position", pct(summary.concentration_top1_pct))


def render_nav_risk(nav_history: pd.DataFrame) -> None:
    if nav_history.empty:
        st.info("No NAV history found yet.")
        return

    nav = nav_history.copy()
    nav["date"] = pd.to_datetime(nav["date"])
    chart = nav.set_index("date")
    left, right = st.columns([1.25, 1])
    with left:
        st.line_chart(chart[["total_net_worth", "equity_peak"]])
    with right:
        st.line_chart(chart[["drawdown"]])

    returns = chart["total_net_worth"].pct_change().dropna()
    if not returns.empty:
        fig = go.Figure()
        fig.add_histogram(x=returns * 100, nbinsx=30, marker_color="#2563EB")
        fig.update_layout(
            height=280,
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis_title="Daily NAV return %",
            yaxis_title="Observations",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_exposure_tables(enriched_positions: pd.DataFrame) -> None:
    broker_table = broker_risk_table(enriched_positions)
    concentration = concentration_table(enriched_positions)

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("Broker Risk")
        st.dataframe(clean_display_frame(broker_table), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Concentration")
        st.dataframe(clean_display_frame(concentration), use_container_width=True, hide_index=True)

    display_cols = [
        "date",
        "broker",
        "ticker",
        "asset_type",
        "shares",
        "current_price",
        "market_value",
        "gross_exposure",
        "signed_delta_exposure",
        "unrealized_pnl",
        "currency",
    ]
    st.subheader("Position Risk Ledger")
    if enriched_positions.empty:
        st.info("No live position rows found yet.")
    else:
        st.dataframe(
            enriched_positions.reindex(columns=display_cols).rename(
                columns={
                    "date": "Date",
                    "broker": "Broker",
                    "ticker": "Ticker",
                    "asset_type": "Asset Type",
                    "shares": "Shares",
                    "current_price": "Current Price",
                    "market_value": "Market Value",
                    "gross_exposure": "Gross Exposure",
                    "signed_delta_exposure": "Delta Exposure",
                    "unrealized_pnl": "Unrealized P&L",
                    "currency": "Currency",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_inverse_hedge_lab(summary, default_budget: float) -> None:
    st.subheader("Beta Hedge")
    hedge_choice = st.selectbox("Hedge asset", ["SQQQ (3x Short QQQ)", "SPXU (3x Short SPY)", "SH (1x Short SPY)"])
    hedge_symbol = hedge_choice.split(" ")[0]
    leverage = -3.0 if "3x" in hedge_choice else -1.0

    history = fetch_price_history(hedge_symbol, "1mo")
    market_price = latest_price_from_history(history)
    atr = average_true_range(history)

    cols = st.columns(4)
    portfolio_value = cols[0].number_input("Portfolio NAV", value=float(summary.latest_nav or summary.gross_exposure or 100000.0), step=500.0)
    portfolio_beta = cols[1].number_input("Portfolio beta", value=float(summary.portfolio_beta or 1.0), step=0.05)
    budget = cols[2].number_input("Hedge budget", value=float(default_budget), step=100.0)
    hedge_price = cols[3].number_input("Hedge price", value=float(market_price or 1.0), step=0.1)

    plan = inverse_hedge_plan(
        portfolio_value=portfolio_value,
        portfolio_beta=portfolio_beta,
        hedge_asset=hedge_symbol,
        hedge_price=hedge_price,
        budget=budget,
        leverage=leverage,
        atr=atr,
    )

    impact = st.columns(5)
    impact[0].metric("Shares", f"{plan.shares_to_buy:.2f}")
    impact[1].metric("Current Delta", money(plan.current_delta))
    impact[2].metric("Hedge Delta", signed_money(plan.hedge_delta))
    impact[3].metric("Beta After", f"{plan.beta_after:.3f}")
    impact[4].metric("Risk Covered", pct(plan.effective_hedge_pct))

    exits = st.columns(3)
    exits[0].metric("Live ATR", money(atr))
    exits[1].metric("Stop Loss", "n/a" if plan.stop_loss is None else money(plan.stop_loss))
    exits[2].metric("Take Profit", "n/a" if plan.take_profit is None else money(plan.take_profit))


def render_manual_hedge_lab(summary) -> None:
    st.subheader("Manual Hedge Check")
    hedge_type = st.selectbox("Strategy type", ["Put Debit Spread", "Outright Long Put", "Short Micro-Futures"])

    with st.form("manual_hedge_form", border=True):
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            underlying = st.text_input("Underlying", value="QQQ").upper().strip()
            contracts = st.number_input("Contracts", min_value=0, value=0)
        with col_2:
            expiry = st.date_input("Expiration", value=date.today() + pd.Timedelta(days=30))
            underlying_price_override = st.number_input("Underlying price override", min_value=0.0, value=0.0)
        with col_3:
            long_strike = st.number_input("Long put strike", min_value=1.0, value=420.0)
            short_strike = (
                st.number_input("Short put strike", min_value=1.0, value=410.0)
                if hedge_type == "Put Debit Spread"
                else 0.0
            )
        submitted = st.form_submit_button("Calculate Protection", use_container_width=True)

    if not submitted:
        return

    history = fetch_price_history(underlying, "3mo")
    spot = underlying_price_override or latest_price_from_history(history)
    if spot <= 0:
        st.error("Could not resolve underlying price.")
        return

    days_to_expiry = max((expiry - date.today()).days, 1)
    beta_exposure = summary.beta_adjusted_exposure or summary.net_delta_exposure
    if hedge_type == "Short Micro-Futures":
        multiplier = micro_future_multiplier(underlying)
        diagnosis = hedge_diagnosis(
            beta_adjusted_exposure=beta_exposure,
            net_contract_delta=-1.0,
            underlying_price=spot,
            contracts=int(contracts),
            contract_size=multiplier,
        )
    else:
        volatility = annualized_vol_from_history(history) or 0.20
        time_to_expiry = days_to_expiry / 365.0
        long_put = black_scholes_greeks(spot, long_strike, time_to_expiry, 0.045, volatility, "put")
        short_delta = 0.0
        if hedge_type == "Put Debit Spread":
            short_put = black_scholes_greeks(spot, short_strike, time_to_expiry, 0.045, volatility, "put")
            short_delta = -short_put["Delta"]
        net_delta = long_put["Delta"] + short_delta
        diagnosis = hedge_diagnosis(
            beta_adjusted_exposure=beta_exposure,
            net_contract_delta=net_delta,
            underlying_price=spot,
            contracts=int(contracts),
        )

    cols = st.columns(4)
    cols[0].metric("Net Delta / Contract", f"{diagnosis.net_contract_delta:.3f}")
    cols[1].metric("Protection", money(diagnosis.total_protection))
    cols[2].metric("Coverage", pct(diagnosis.coverage_pct))
    cols[3].metric("More Contracts", str(diagnosis.additional_contracts_needed))

    if diagnosis.verdict == "properly_hedged":
        st.success("Verdict: properly hedged.")
    elif diagnosis.verdict == "partially_hedged":
        st.warning(f"Verdict: partially hedged. Remaining exposure: {money(diagnosis.unhedged_exposure)}.")
    elif diagnosis.verdict == "severely_under_hedged":
        st.error(f"Verdict: severely under-hedged. Remaining exposure: {money(diagnosis.unhedged_exposure)}.")
    else:
        st.error(f"Verdict: unhedged. Remaining exposure: {money(diagnosis.unhedged_exposure)}.")


ledger_path = default_portfolio_ledger_path()
positions = load_latest_live_positions(ledger_path)
enriched_positions = enrich_position_risk(positions)
nav_history = compute_nav_drawdowns(load_historical_nav(ledger_path))
summary = summarize_portfolio_risk(enriched_positions, nav_history)
ibkr_metrics = read_json(DEFAULT_IBKR_METRICS_PATH)
default_cash = safe_float(ibkr_metrics.get("Available_Cash_USD"), summary.latest_cash)

st.title("Risk Management")
st.caption("Portfolio risk, drawdowns, concentration, and hedge sizing from the runtime ledger.")
render_metric_grid(summary)

tab_exposure, tab_nav, tab_hedge = st.tabs(["Exposure", "NAV Risk", "Hedge Lab"])

with tab_exposure:
    render_exposure_tables(enriched_positions)

with tab_nav:
    render_nav_risk(nav_history)

with tab_hedge:
    render_inverse_hedge_lab(summary, default_budget=max(default_cash, 0.0))
    st.divider()
    render_manual_hedge_lab(summary)

st.caption(f"Ledger database: `{ledger_path.relative_to(REPO_ROOT)}`")
