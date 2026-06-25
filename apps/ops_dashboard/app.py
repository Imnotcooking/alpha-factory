"""Operations dashboard for live-account read-only monitoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.broker_monitor import (  # noqa: E402
    connect_readonly_snapshot,
    render_account_metrics,
    render_broker_health_json,
    render_cash_table,
    render_open_orders_table,
    render_positions_table,
    yes_no,
)
from oqp.brokers import (  # noqa: E402
    BrokerProfileError,
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import load_settings  # noqa: E402
from oqp.portfolio import (  # noqa: E402
    compute_nav_drawdowns,
    default_portfolio_ledger_path,
    load_historical_nav,
    load_latest_live_positions,
)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


st.set_page_config(page_title="Ops Monitor", layout="wide", page_icon="OPS")

settings = load_settings()

st.title("Ops Monitor")
st.caption("Live IBKR account monitor. Read-only by design.")
st.error("Live order placement is unavailable from this dashboard.")

ledger_path = default_portfolio_ledger_path()
ledger_df = load_latest_live_positions(ledger_path)
nav_df = compute_nav_drawdowns(load_historical_nav(ledger_path))

st.subheader("Unified Portfolio Ledger")
if ledger_df.empty:
    st.info(
        "No unified portfolio snapshot found yet. Run the live portfolio "
        "snapshot job to populate the shared live_positions ledger."
    )
else:
    ledger_cols = st.columns(4)
    ledger_cols[0].metric("Snapshot Date", str(ledger_df["date"].iloc[0]))
    ledger_cols[1].metric("Ledger Rows", str(len(ledger_df)))
    ledger_cols[2].metric("Brokers", str(ledger_df["broker"].nunique()))
    ledger_cols[3].metric("Options", str(ledger_df["asset_type"].eq("Option").sum()))
    st.dataframe(ledger_df, use_container_width=True, hide_index=True)
st.caption(f"Ledger database: {display_path(ledger_path)}")

st.subheader("Portfolio Equity History")
if nav_df.empty:
    st.info(
        "No NAV history found yet. Run the portfolio snapshot job after "
        "positions are populated to write the first NAV row."
    )
else:
    latest_nav = nav_df.iloc[-1]
    max_drawdown = float(nav_df["drawdown"].min())
    max_drawdown_pct = float(nav_df["drawdown_pct"].min()) * 100
    nav_cols = st.columns(4)
    nav_cols[0].metric("NAV Days", str(len(nav_df)))
    nav_cols[1].metric("Latest NAV", f"{float(latest_nav['total_net_worth']):,.2f}")
    nav_cols[2].metric("Daily PnL", f"{float(latest_nav['daily_pnl']):,.2f}")
    nav_cols[3].metric(
        "Max Drawdown",
        f"{max_drawdown:,.2f}",
        f"{max_drawdown_pct:.2f}%",
    )
    nav_chart = nav_df.set_index("date")[["total_net_worth", "equity_peak"]]
    drawdown_chart = nav_df.set_index("date")[["drawdown"]]
    st.line_chart(nav_chart)
    st.line_chart(drawdown_chart)

if not settings.ibkr_live_monitor_enabled:
    st.warning(
        "Live read-only monitoring is disabled. Set "
        "IBKR_LIVE_MONITOR_ENABLED=true only on a secured server/session."
    )
    st.stop()

try:
    broker_config = get_broker_profile_config("ibkr_live_readonly", settings=settings)
except BrokerProfileError as exc:
    st.warning(str(exc))
    st.stop()

broker = get_broker_adapter("ibkr", settings=settings)
snapshot = connect_readonly_snapshot(broker, broker_config)
broker_health = snapshot["health"]
account_summary = snapshot["account_summary"]
cash_balances = snapshot["cash_balances"]
positions = snapshot["positions"]
open_orders = snapshot["open_orders"]
snapshot_error = snapshot["snapshot_error"]

summary_cols = st.columns(4)
summary_cols[0].metric("Profile", broker_config.metadata.get("profile", ""))
summary_cols[1].metric("IBKR", broker_health.status.value)
summary_cols[2].metric("Snapshot", "ready" if account_summary else "offline")
summary_cols[3].metric("Positions", str(len(positions)))

if account_summary:
    render_account_metrics(account_summary)
elif snapshot_error:
    st.warning(f"Connected to IBKR, but snapshot fetch failed: {snapshot_error}")
else:
    st.info(
        "Log in to live TWS or IB Gateway on this host and enable API socket access. "
        "The dashboard connects read-only to the local socket."
    )

left, right = st.columns([1.4, 1])

with left:
    st.subheader("Live Read-Only Positions")
    render_positions_table(positions)

with right:
    st.subheader("Cash Balances")
    render_cash_table(cash_balances)

st.subheader("Open Orders")
render_open_orders_table(open_orders)

st.subheader("Runtime Guardrails")
guardrails_df = pd.DataFrame(
    [
        {"Check": "Live monitor enabled", "Value": yes_no(settings.ibkr_live_monitor_enabled)},
        {"Check": "Live trading allowed", "Value": yes_no(settings.allow_live_trading)},
        {"Check": "Broker read-only", "Value": yes_no(broker_config.readonly)},
        {"Check": "Host", "Value": broker_config.host},
        {"Check": "Port", "Value": str(broker_config.port)},
        {"Check": "Client ID", "Value": str(broker_config.client_id)},
    ]
)
st.dataframe(guardrails_df, use_container_width=True, hide_index=True)

with st.expander("Broker Health Payload", expanded=False):
    render_broker_health_json(broker_health)
