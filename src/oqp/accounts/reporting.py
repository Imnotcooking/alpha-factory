"""Presentation-friendly account ledger transforms."""

from __future__ import annotations

import pandas as pd


def account_nav_drawdowns(nav_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "net_liquidation",
        "cash",
        "daily_pnl",
        "position_count",
        "equity_peak",
        "drawdown",
        "drawdown_pct",
    ]
    if nav_history.empty:
        return pd.DataFrame(columns=columns)

    out = nav_history.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in ("net_liquidation", "cash", "daily_pnl"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["position_count"] = pd.to_numeric(
        out["position_count"],
        errors="coerce",
    ).fillna(0)
    out["equity_peak"] = out["net_liquidation"].cummax()
    out["drawdown"] = out["net_liquidation"] - out["equity_peak"]
    out["drawdown_pct"] = (
        out["drawdown"] / out["equity_peak"].replace(0, pd.NA)
    ).fillna(0.0)
    return out.reindex(columns=columns)


def account_positions_display(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Asset Class",
                "Quantity",
                "Market Price",
                "Market Value",
                "Unrealized P&L",
                "Currency",
                "As Of",
            ]
        )

    columns = [
        "symbol",
        "asset_class",
        "quantity",
        "market_price",
        "market_value",
        "unrealized_pnl",
        "currency",
        "as_of",
    ]
    return positions.reindex(columns=columns).rename(
        columns={
            "symbol": "Symbol",
            "asset_class": "Asset Class",
            "quantity": "Quantity",
            "market_price": "Market Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L",
            "currency": "Currency",
            "as_of": "As Of",
        }
    )


def account_asset_summary(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(
            columns=["Asset Class", "Rows", "Market Value", "Unrealized P&L"]
        )

    out = positions.copy()
    for column in ("market_value", "unrealized_pnl"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return (
        out.groupby("asset_class")
        .agg(
            rows=("symbol", "count"),
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "asset_class": "Asset Class",
                "rows": "Rows",
                "market_value": "Market Value",
                "unrealized_pnl": "Unrealized P&L",
            }
        )
    )


def account_trade_events_display(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Occurred",
        "Environment",
        "Event",
        "Symbol",
        "Side",
        "Quantity",
        "Price",
        "Strategy",
        "Order / Proposal",
        "Broker Order",
        "Currency",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    display = events.copy()
    for column in ("quantity", "price", "commission"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce")
    rename = {
        "occurred_at": "Occurred",
        "environment": "Environment",
        "event_type": "Event",
        "symbol": "Symbol",
        "side": "Side",
        "quantity": "Quantity",
        "price": "Price",
        "strategy_id": "Strategy",
        "order_id": "Order / Proposal",
        "broker_order_id": "Broker Order",
        "currency": "Currency",
    }
    return (
        display.reindex(columns=list(rename))
        .rename(columns=rename)
        .reindex(columns=columns)
    )


def account_trade_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["Event", "Rows", "Symbols", "Quantity"])

    out = events.copy()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(0.0)
    return (
        out.groupby("event_type")
        .agg(
            rows=("event_id", "count"),
            symbols=("symbol", lambda values: ", ".join(sorted(set(map(str, values))))),
            quantity=("quantity", "sum"),
        )
        .reset_index()
        .rename(
            columns={
                "event_type": "Event",
                "rows": "Rows",
                "symbols": "Symbols",
                "quantity": "Quantity",
            }
        )
    )
