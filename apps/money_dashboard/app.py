"""Money dashboard command center for real portfolio monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.portfolio import (  # noqa: E402
    DEFAULT_BANKED_PROFITS_PATH,
    DEFAULT_IBKR_METRICS_PATH,
    DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH,
    compute_nav_drawdowns,
    default_portfolio_ledger_path,
    load_historical_nav,
    load_latest_live_positions,
)


st.set_page_config(page_title="Money Dashboard", layout="wide", page_icon="USD")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def money(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:,.2f}"


def signed_money(value: float | None) -> str:
    if value is None:
        return "missing"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.2f}"


def percent(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.2f}%"


def position_multiplier(asset_type: Any) -> float:
    return 100.0 if str(asset_type).strip().lower() == "option" else 1.0


def enrich_positions(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=[
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
                "delta",
                "delta_exposure",
            ]
        )

    out = positions.copy()
    for column in ("shares", "avg_cost", "current_price", "unrealized_pnl", "delta"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["multiplier"] = out["asset_type"].map(position_multiplier)
    out["market_value"] = out["shares"] * out["current_price"] * out["multiplier"]
    fallback_pnl = out["market_value"] - (out["shares"] * out["avg_cost"] * out["multiplier"])
    out["unrealized_pnl"] = out["unrealized_pnl"].where(
        out["unrealized_pnl"].ne(0.0),
        fallback_pnl,
    )
    out["delta_exposure"] = out["market_value"] * out["delta"].abs()
    return out


def broker_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame(
            columns=["Broker", "Rows", "Market Value", "Delta Exposure", "Unrealized P&L"]
        )
    summary = (
        enriched.groupby("broker")
        .agg(
            rows=("ticker", "count"),
            market_value=("market_value", "sum"),
            delta_exposure=("delta_exposure", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "broker": "Broker",
                "rows": "Rows",
                "market_value": "Market Value",
                "delta_exposure": "Delta Exposure",
                "unrealized_pnl": "Unrealized P&L",
            }
        )
    )
    return summary


def asset_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return pd.DataFrame(columns=["Asset Type", "Rows", "Market Value", "Delta Exposure"])
    return (
        enriched.groupby("asset_type")
        .agg(
            rows=("ticker", "count"),
            market_value=("market_value", "sum"),
            delta_exposure=("delta_exposure", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "asset_type": "Asset Type",
                "rows": "Rows",
                "market_value": "Market Value",
                "delta_exposure": "Delta Exposure",
            }
        )
    )


def display_positions(enriched: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "delta",
        "delta_exposure",
    ]
    if enriched.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Broker",
                "Ticker",
                "Asset Type",
                "Shares",
                "Avg Cost",
                "Current Price",
                "Market Value",
                "Unrealized P&L",
                "Currency",
                "Delta",
                "Delta Exposure",
            ]
        )
    return enriched.reindex(columns=columns).rename(
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
            "delta": "Delta",
            "delta_exposure": "Delta Exposure",
        }
    )


def state_table(*, ibkr_metrics: dict[str, Any], manual_inputs: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Item": "Portfolio ledger",
                "Value": display_path(default_portfolio_ledger_path()),
            },
            {
                "Item": "IBKR metrics",
                "Value": display_path(DEFAULT_IBKR_METRICS_PATH),
            },
            {
                "Item": "Manual inputs",
                "Value": display_path(DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH),
            },
            {
                "Item": "Banked profits",
                "Value": display_path(DEFAULT_BANKED_PROFITS_PATH),
            },
            {
                "Item": "IBKR cash",
                "Value": money(as_float(ibkr_metrics.get("Available_Cash_USD")))
                if ibkr_metrics
                else "missing",
            },
            {
                "Item": "Manual input fields",
                "Value": str(len(manual_inputs)),
            },
        ]
    )


ledger_path = default_portfolio_ledger_path()
positions = load_latest_live_positions(ledger_path)
enriched_positions = enrich_positions(positions)
nav_history = compute_nav_drawdowns(load_historical_nav(ledger_path))
ibkr_metrics = read_json(DEFAULT_IBKR_METRICS_PATH)
manual_inputs = read_json(DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH)
banked_profits = read_json(DEFAULT_BANKED_PROFITS_PATH)

latest_nav = None if nav_history.empty else nav_history.iloc[-1]
latest_date = "missing" if latest_nav is None else latest_nav["date"].date().isoformat()
latest_nav_value = None if latest_nav is None else as_float(latest_nav["total_net_worth"])
daily_pnl = None if latest_nav is None else as_float(latest_nav["daily_pnl"])
latest_cash = None if latest_nav is None else as_float(latest_nav["total_cash"])
portfolio_beta = None if latest_nav is None else as_float(latest_nav["portfolio_beta"])
max_drawdown = None if nav_history.empty else as_float(nav_history["drawdown"].min())
max_drawdown_pct = None if nav_history.empty else as_float(nav_history["drawdown_pct"].min()) * 100

st.title("Money Dashboard")
st.caption("Real portfolio command center backed by the shared runtime ledger.")

metric_cols = st.columns(5)
metric_cols[0].metric("Latest NAV", money(latest_nav_value))
metric_cols[1].metric("Daily P&L", signed_money(daily_pnl))
metric_cols[2].metric("Cash", money(latest_cash))
metric_cols[3].metric("Portfolio Beta", "missing" if portfolio_beta is None else f"{portfolio_beta:.4f}")
metric_cols[4].metric("Snapshot Date", latest_date)

drawdown_cols = st.columns(4)
drawdown_cols[0].metric("Stored NAV Days", str(len(nav_history)))
drawdown_cols[1].metric("Max Drawdown", signed_money(max_drawdown))
drawdown_cols[2].metric("Max Drawdown %", percent(max_drawdown_pct))
drawdown_cols[3].metric("Ledger Rows", str(len(positions)))

if nav_history.empty:
    st.info("No stored NAV history found yet.")
else:
    st.subheader("Equity & Drawdown")
    nav_chart = nav_history.set_index("date")[["total_net_worth", "equity_peak"]]
    drawdown_chart = nav_history.set_index("date")[["drawdown"]]
    chart_left, chart_right = st.columns([1.2, 1])
    with chart_left:
        st.line_chart(nav_chart)
    with chart_right:
        st.line_chart(drawdown_chart)

st.subheader("Portfolio Ledger")
if enriched_positions.empty:
    st.info("No live position rows found yet.")
else:
    st.dataframe(display_positions(enriched_positions), use_container_width=True, hide_index=True)
st.caption(f"Ledger database: {display_path(ledger_path)}")

summary_left, summary_right = st.columns([1, 1])
with summary_left:
    st.subheader("Broker Allocation")
    st.dataframe(broker_summary(enriched_positions), use_container_width=True, hide_index=True)

with summary_right:
    st.subheader("Asset Mix")
    st.dataframe(asset_summary(enriched_positions), use_container_width=True, hide_index=True)

state_left, state_right = st.columns([1, 1])
with state_left:
    st.subheader("Runtime State")
    st.dataframe(
        state_table(ibkr_metrics=ibkr_metrics, manual_inputs=manual_inputs),
        use_container_width=True,
        hide_index=True,
    )

with state_right:
    st.subheader("Cash & Real Asset Inputs")
    cash_rows = [
        {"Source": "IBKR Available Cash", "Value": money(as_float(ibkr_metrics.get("Available_Cash_USD")))},
        {"Source": "IBKR Total NAV", "Value": money(as_float(ibkr_metrics.get("Total_NAV_USD")))},
        {"Source": "IBKR Margin Buffer", "Value": money(as_float(ibkr_metrics.get("Margin_Buffer_USD")))},
        {"Source": "Trading212 Banked EUR", "Value": money(as_float(banked_profits.get("Trading212_EUR")))},
        {"Source": "Futubull Banked USD", "Value": money(as_float(banked_profits.get("Futubull_USD")))},
    ]
    st.dataframe(pd.DataFrame(cash_rows), use_container_width=True, hide_index=True)
