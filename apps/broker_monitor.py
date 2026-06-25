"""Shared Streamlit helpers for broker read-only dashboards."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from oqp.brokers import BrokerConnectionStatus


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def money(value: float | None, currency: str = "USD") -> str:
    return f"{currency} {value:,.2f}" if value is not None else "n/a"


def connect_readonly_snapshot(broker: Any, broker_config: Any) -> dict[str, Any]:
    broker_health = broker.connect(broker_config)
    broker_connected = broker_health.status == BrokerConnectionStatus.CONNECTED

    account_summary = None
    cash_balances = []
    positions = []
    open_orders = []
    snapshot_error = None

    if broker_connected:
        try:
            account_summary = broker.get_account_summary()
            cash_balances = list(broker.get_cash_balances())
            positions = list(broker.get_positions())
            open_orders = list(broker.get_open_orders())
        except Exception as exc:
            snapshot_error = str(exc)

    return {
        "health": broker_health,
        "connected": broker_connected,
        "account_summary": account_summary,
        "cash_balances": cash_balances,
        "positions": positions,
        "open_orders": open_orders,
        "snapshot_error": snapshot_error,
    }


def render_account_metrics(account_summary: Any | None) -> None:
    if account_summary is None:
        return

    acct_cols = st.columns(4)
    acct_cols[0].metric("Account", account_summary.account_id)
    acct_cols[1].metric(
        "Net Liquidation",
        money(account_summary.net_liquidation, account_summary.currency),
    )
    acct_cols[2].metric("Cash", money(account_summary.cash, account_summary.currency))
    acct_cols[3].metric(
        "Buying Power",
        money(account_summary.buying_power, account_summary.currency),
    )


def render_positions_table(positions: list[Any]) -> None:
    position_rows = [
        {
            "Symbol": position.instrument.symbol,
            "Asset": position.instrument.asset_class.value,
            "Quantity": position.quantity,
            "Average Cost": position.average_cost,
            "Market Price": position.market_price,
            "Market Value": position.market_value,
            "Unrealized PnL": position.unrealized_pnl,
        }
        for position in positions
    ]
    st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)


def render_cash_table(cash_balances: list[Any]) -> None:
    cash_rows = [
        {
            "Currency": balance.currency,
            "Cash": balance.cash,
            "Settled Cash": balance.settled_cash,
            "Buying Power": balance.buying_power,
        }
        for balance in cash_balances
    ]
    st.dataframe(pd.DataFrame(cash_rows), use_container_width=True, hide_index=True)


def render_open_orders_table(open_orders: list[Any]) -> None:
    order_rows = [
        {
            "Broker Order ID": receipt.broker_order_id,
            "Symbol": receipt.order.instrument.symbol,
            "Side": receipt.order.side.value,
            "Quantity": receipt.order.quantity,
            "Type": receipt.order.order_type.value,
            "Status": receipt.status.value,
            "Limit": receipt.order.limit_price,
            "Stop": receipt.order.stop_price,
        }
        for receipt in open_orders
    ]
    st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)


def render_broker_health_json(broker_health: Any) -> None:
    broker_payload = asdict(broker_health)
    broker_payload["checked_at"] = broker_health.checked_at.isoformat()
    st.json(broker_payload)
