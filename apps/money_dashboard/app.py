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

from oqp.accounts import (  # noqa: E402
    account_asset_summary,
    account_nav_drawdowns,
    account_positions_display,
    account_trade_event_summary,
    account_trade_events_display,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_trade_events,
    load_latest_account_nav,
    load_latest_account_positions,
)
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


def state_table(
    *,
    account_ledger_path: Path,
    live_account_nav: pd.DataFrame,
    live_account_positions: pd.DataFrame,
    ibkr_metrics: dict[str, Any],
    manual_inputs: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Item": "Unified account ledger",
                "Value": display_path(account_ledger_path),
            },
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
                "Item": "Unified live NAV rows",
                "Value": str(len(live_account_nav)),
            },
            {
                "Item": "Unified live positions",
                "Value": str(len(live_account_positions)),
            },
            {
                "Item": "Manual input fields",
                "Value": str(len(manual_inputs)),
            },
        ]
    )


account_ledger_path = default_account_ledger_path()
live_account_nav = load_latest_account_nav(account_ledger_path, environment="live")
live_account_nav_history = account_nav_drawdowns(
    load_account_nav_history(account_ledger_path, environment="live")
)
live_account_positions = load_latest_account_positions(
    account_ledger_path,
    environment="live",
)
live_account_events = load_account_trade_events(
    account_ledger_path,
    environment="live",
    limit=25,
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

latest_account_nav = None if live_account_nav.empty else live_account_nav.iloc[0]
account_nav_value = (
    None
    if latest_account_nav is None
    else as_float(latest_account_nav.get("net_liquidation"))
)
account_daily_pnl = (
    None if latest_account_nav is None else as_float(latest_account_nav.get("daily_pnl"))
)
account_cash = None if latest_account_nav is None else as_float(latest_account_nav.get("cash"))
account_position_count = (
    None
    if latest_account_nav is None
    else int(as_float(latest_account_nav.get("position_count")))
)
account_snapshot_as_of = (
    None if latest_account_nav is None else str(latest_account_nav.get("as_of") or "")
)

dashboard_nav_value = account_nav_value if account_nav_value is not None else latest_nav_value
dashboard_daily_pnl = account_daily_pnl if account_daily_pnl is not None else daily_pnl
dashboard_cash = account_cash if account_cash is not None else latest_cash
dashboard_position_count = (
    account_position_count if account_position_count is not None else len(positions)
)
dashboard_snapshot = account_snapshot_as_of or latest_date
dashboard_source = "unified account ledger" if latest_account_nav is not None else "portfolio ledger"

if not live_account_nav_history.empty:
    max_drawdown = as_float(live_account_nav_history["drawdown"].min())
    max_drawdown_pct = as_float(live_account_nav_history["drawdown_pct"].min()) * 100

st.title("Money Dashboard")
st.caption("Real portfolio command center backed by the unified account ledger.")

metric_cols = st.columns(6)
metric_cols[0].metric("Latest NAV", money(dashboard_nav_value))
metric_cols[1].metric("Daily P&L", signed_money(dashboard_daily_pnl))
metric_cols[2].metric("Cash", money(dashboard_cash))
metric_cols[3].metric("Positions", str(dashboard_position_count))
metric_cols[4].metric("Snapshot", dashboard_snapshot)
metric_cols[5].metric("Source", dashboard_source)

drawdown_cols = st.columns(4)
drawdown_cols[0].metric(
    "Stored NAV Days",
    str(len(live_account_nav_history) if not live_account_nav_history.empty else len(nav_history)),
)
drawdown_cols[1].metric("Max Drawdown", signed_money(max_drawdown))
drawdown_cols[2].metric("Max Drawdown %", percent(max_drawdown_pct))
drawdown_cols[3].metric("Ledger Rows", str(len(live_account_positions) or len(positions)))

if not live_account_nav_history.empty:
    st.subheader("Unified Account Equity & Drawdown")
    account_chart = live_account_nav_history.set_index("date")[
        ["net_liquidation", "equity_peak"]
    ]
    account_drawdown_chart = live_account_nav_history.set_index("date")[["drawdown"]]
    chart_left, chart_right = st.columns([1.2, 1])
    with chart_left:
        st.line_chart(account_chart)
    with chart_right:
        st.line_chart(account_drawdown_chart)
elif nav_history.empty:
    st.info("No stored NAV history found yet.")
else:
    st.subheader("Portfolio Ledger Equity & Drawdown")
    nav_chart = nav_history.set_index("date")[["total_net_worth", "equity_peak"]]
    drawdown_chart = nav_history.set_index("date")[["drawdown"]]
    chart_left, chart_right = st.columns([1.2, 1])
    with chart_left:
        st.line_chart(nav_chart)
    with chart_right:
        st.line_chart(drawdown_chart)

st.subheader("Unified Live Account Positions")
account_positions_table = account_positions_display(live_account_positions)
if account_positions_table.empty:
    st.info("No unified live account position rows found yet.")
else:
    st.dataframe(account_positions_table, use_container_width=True, hide_index=True)
st.caption(f"Account ledger database: {display_path(account_ledger_path)}")

account_left, account_right = st.columns([1, 1])
with account_left:
    st.subheader("Unified Asset Mix")
    st.dataframe(
        account_asset_summary(live_account_positions),
        use_container_width=True,
        hide_index=True,
    )
with account_right:
    st.subheader("Unified Snapshot State")
    account_state_rows = [
        {
            "Item": "Account",
            "Value": (
                "missing"
                if latest_account_nav is None
                else str(latest_account_nav.get("account_id") or "missing")
            ),
        },
        {
            "Item": "Profile",
            "Value": (
                "missing"
                if latest_account_nav is None
                else str(latest_account_nav.get("profile") or "missing")
            ),
        },
        {
            "Item": "Environment",
            "Value": (
                "missing"
                if latest_account_nav is None
                else str(latest_account_nav.get("environment") or "missing")
            ),
        },
        {"Item": "As Of", "Value": dashboard_snapshot},
    ]
    st.dataframe(pd.DataFrame(account_state_rows), use_container_width=True, hide_index=True)

st.subheader("Unified Live Trade Events")
if live_account_events.empty:
    st.info("No live account trade events have been recorded yet.")
else:
    events_left, events_right = st.columns([1.3, 1])
    with events_left:
        st.dataframe(
            account_trade_events_display(live_account_events),
            use_container_width=True,
            hide_index=True,
        )
    with events_right:
        st.dataframe(
            account_trade_event_summary(live_account_events),
            use_container_width=True,
            hide_index=True,
        )

st.subheader("Legacy Portfolio Ledger")
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
        state_table(
            account_ledger_path=account_ledger_path,
            live_account_nav=live_account_nav,
            live_account_positions=live_account_positions,
            ibkr_metrics=ibkr_metrics,
            manual_inputs=manual_inputs,
        ),
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
