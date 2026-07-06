"""Presentation-friendly account ledger transforms."""

from __future__ import annotations

import pandas as pd


def blended_live_nav_history(
    unified_nav: pd.DataFrame,
    broker_nav: pd.DataFrame,
    *,
    manual_usd_value: float | None = None,
    manual_usd_cash: float | None = None,
) -> pd.DataFrame:
    """Return a chart-ready live NAV series while unified history warms up.

    ``unified_live`` is the authoritative series once it exists. During the
    transition from broker-only monitoring to broker-plus-manual monitoring,
    earlier broker rows are adjusted by the current manual USD value so the
    overview chart does not collapse into a one-point axis.
    """

    columns = [
        "date",
        "account_key",
        "account_id",
        "broker",
        "profile",
        "environment",
        "as_of",
        "currency",
        "net_liquidation",
        "cash",
        "daily_pnl",
        "position_count",
        "snapshot_id",
    ]
    unified = pd.DataFrame() if unified_nav is None else unified_nav.copy()
    broker = pd.DataFrame() if broker_nav is None else broker_nav.copy()
    if unified.empty and broker.empty:
        return pd.DataFrame(columns=columns)

    manual_value = _number(manual_usd_value)
    manual_cash = _number(manual_usd_cash)
    frames: list[pd.DataFrame] = []

    if not broker.empty:
        broker = broker.reindex(columns=columns).copy()
        broker["date"] = pd.to_datetime(broker["date"], errors="coerce")
        if not unified.empty and "date" in unified:
            first_unified_date = pd.to_datetime(unified["date"], errors="coerce").min()
            broker = broker.loc[broker["date"].lt(first_unified_date)].copy()
        if not broker.empty:
            for column in ("net_liquidation", "cash", "position_count"):
                broker[column] = pd.to_numeric(broker[column], errors="coerce").fillna(0.0)
            broker["net_liquidation"] = broker["net_liquidation"] + manual_value
            broker["cash"] = broker["cash"] + manual_cash
            broker["profile"] = "unified_live"
            broker["broker"] = "unified"
            broker["account_id"] = "unified_live"
            broker["account_key"] = "live:unified:unified_live:unified_live"
            broker["snapshot_id"] = "synthetic-" + broker["date"].dt.strftime("%Y-%m-%d")
            frames.append(broker)

    if not unified.empty:
        frames.append(unified.reindex(columns=columns).copy())

    if not frames:
        return pd.DataFrame(columns=columns)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce", utc=True)
    for column in ("net_liquidation", "cash", "position_count"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out = (
        out.dropna(subset=["date"])
        .sort_values(["date", "as_of"])
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    out["daily_pnl"] = out["net_liquidation"].diff().fillna(0.0)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["as_of"] = out["as_of"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out.reindex(columns=columns)


def account_nav_drawdowns(nav_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "net_liquidation",
        "cash",
        "daily_pnl",
        "daily_return",
        "cumulative_return",
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
    previous_nav = out["net_liquidation"].shift(1)
    out["daily_return"] = (
        out["daily_pnl"] / previous_nav.replace(0, pd.NA)
    ).fillna(0.0)
    first_nav = out["net_liquidation"].replace(0, pd.NA).dropna()
    base_nav = None if first_nav.empty else float(first_nav.iloc[0])
    out["cumulative_return"] = (
        0.0
        if base_nav is None
        else (out["net_liquidation"] / base_nav - 1.0).fillna(0.0)
    )
    out["equity_peak"] = out["net_liquidation"].cummax()
    out["drawdown"] = out["net_liquidation"] - out["equity_peak"]
    out["drawdown_pct"] = (
        out["drawdown"] / out["equity_peak"].replace(0, pd.NA)
    ).fillna(0.0)
    return out.reindex(columns=columns)


def _number(value: float | int | str | None) -> float:
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(parsed) else parsed


def account_position_history_by_symbol(positions: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "symbol", "market_value", "quantity", "unrealized_pnl"]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    out = positions.copy()
    if "snapshot_date" not in out.columns:
        return pd.DataFrame(columns=columns)
    out["date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    for column in ("market_value", "quantity", "unrealized_pnl"):
        if column not in out.columns:
            out[column] = 0.0
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    grouped = (
        out.groupby(["date", "symbol"], dropna=False)
        .agg(
            market_value=("market_value", "sum"),
            quantity=("quantity", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=columns)


def account_position_history_by_asset(positions: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "asset_class", "market_value", "unrealized_pnl"]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    out = positions.copy()
    if "snapshot_date" not in out.columns or "asset_class" not in out.columns:
        return pd.DataFrame(columns=columns)
    out["date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    for column in ("market_value", "unrealized_pnl"):
        if column not in out.columns:
            out[column] = 0.0
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    grouped = (
        out.groupby(["date", "asset_class"], dropna=False)
        .agg(
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=columns)


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


def account_position_totals(positions: pd.DataFrame) -> dict[str, float | int | None]:
    """Aggregate current position-level exposure and P&L for dashboard metrics."""

    if positions.empty:
        return {
            "position_rows": 0,
            "gross_exposure": None,
            "unrealized_pnl": None,
            "realized_pnl": None,
            "total_pnl": None,
        }

    out = positions.copy()
    for column in ("market_value", "unrealized_pnl", "realized_pnl"):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")

    gross_exposure = out["market_value"].abs().sum()
    unrealized = out["unrealized_pnl"].sum(min_count=1)
    realized = out["realized_pnl"].sum(min_count=1)
    total = (
        (0.0 if pd.isna(unrealized) else float(unrealized))
        + (0.0 if pd.isna(realized) else float(realized))
        if not pd.isna(unrealized) or not pd.isna(realized)
        else None
    )
    return {
        "position_rows": int(len(out)),
        "gross_exposure": None if pd.isna(gross_exposure) else float(gross_exposure),
        "unrealized_pnl": None if pd.isna(unrealized) else float(unrealized),
        "realized_pnl": None if pd.isna(realized) else float(realized),
        "total_pnl": total,
    }


def account_profit_breakdown(
    positions: pd.DataFrame,
    *,
    daily_pnl: float | None = None,
) -> pd.DataFrame:
    columns = ["Bucket", "Value"]
    totals = account_position_totals(positions)
    return pd.DataFrame(
        [
            {"Bucket": "Daily P&L", "Value": daily_pnl or 0.0},
            {"Bucket": "Unrealized P&L", "Value": totals["unrealized_pnl"] or 0.0},
            {"Bucket": "Realized P&L", "Value": totals["realized_pnl"] or 0.0},
        ],
        columns=columns,
    )


def account_performance_summary(
    nav_history: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    current_nav: float | None = None,
    current_cash: float | None = None,
    current_daily_pnl: float | None = None,
) -> dict[str, float | int | None]:
    """Summarize account return, drawdown, cash, exposure, and position P&L."""

    drawdowns = account_nav_drawdowns(nav_history)
    latest_nav_row = None if drawdowns.empty else drawdowns.iloc[-1]
    totals = account_position_totals(positions)

    nav = (
        current_nav
        if current_nav is not None
        else None
        if latest_nav_row is None
        else float(latest_nav_row["net_liquidation"])
    )
    cash = (
        current_cash
        if current_cash is not None
        else None
        if latest_nav_row is None
        else float(latest_nav_row["cash"])
    )
    daily_pnl = (
        current_daily_pnl
        if current_daily_pnl is not None
        else None
        if latest_nav_row is None
        else float(latest_nav_row["daily_pnl"])
    )
    latest_return = (
        None if latest_nav_row is None else float(latest_nav_row["daily_return"])
    )
    cumulative_return = (
        None if latest_nav_row is None else float(latest_nav_row["cumulative_return"])
    )
    max_drawdown_pct = (
        None if drawdowns.empty else float(drawdowns["drawdown_pct"].min())
    )
    gross_exposure = totals["gross_exposure"]
    gross_exposure_pct = (
        None
        if nav in (None, 0) or gross_exposure is None
        else float(gross_exposure) / float(nav)
    )
    cash_pct = None if nav in (None, 0) or cash is None else float(cash) / float(nav)

    return {
        "nav_observations": int(len(drawdowns)),
        "latest_nav": nav,
        "latest_cash": cash,
        "cash_pct": cash_pct,
        "daily_pnl": daily_pnl,
        "daily_return": latest_return,
        "cumulative_return": cumulative_return,
        "max_drawdown_pct": max_drawdown_pct,
        "position_rows": int(totals["position_rows"] or 0),
        "gross_exposure": gross_exposure,
        "gross_exposure_pct": gross_exposure_pct,
        "unrealized_pnl": totals["unrealized_pnl"],
        "realized_pnl": totals["realized_pnl"],
        "total_position_pnl": totals["total_pnl"],
    }


def account_top_positions(
    positions: pd.DataFrame,
    *,
    limit: int = 12,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "asset_class",
        "market_value",
        "unrealized_pnl",
        "realized_pnl",
    ]
    if positions.empty or "symbol" not in positions.columns:
        return pd.DataFrame(columns=columns)

    out = positions.copy()
    if "asset_class" not in out.columns:
        out["asset_class"] = ""
    for column in ("market_value", "unrealized_pnl", "realized_pnl"):
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["abs_market_value"] = out["market_value"].abs()
    return (
        out.sort_values("abs_market_value", ascending=False)
        .head(max(int(limit), 1))
        .reindex(columns=columns)
    )


def account_symbol_exposure_pivot(
    symbol_history: pd.DataFrame,
    *,
    limit: int = 8,
) -> pd.DataFrame:
    if symbol_history.empty:
        return pd.DataFrame()
    required = {"date", "symbol", "market_value"}
    if not required.issubset(symbol_history.columns):
        return pd.DataFrame()

    history = symbol_history.copy()
    history["market_value"] = pd.to_numeric(
        history["market_value"],
        errors="coerce",
    ).fillna(0.0)
    latest_date = history["date"].max()
    latest = history[history["date"].eq(latest_date)].copy()
    latest["abs_value"] = latest["market_value"].abs()
    top_symbols = latest.sort_values("abs_value", ascending=False)["symbol"].head(
        max(int(limit), 1)
    )
    filtered = history[history["symbol"].isin(top_symbols)].copy()
    if filtered.empty:
        return pd.DataFrame()
    return (
        filtered.pivot_table(
            index="date",
            columns="symbol",
            values="market_value",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )


def account_asset_exposure_pivot(asset_history: pd.DataFrame) -> pd.DataFrame:
    if asset_history.empty:
        return pd.DataFrame()
    required = {"date", "asset_class", "market_value"}
    if not required.issubset(asset_history.columns):
        return pd.DataFrame()

    history = asset_history.copy()
    history["market_value"] = pd.to_numeric(
        history["market_value"],
        errors="coerce",
    ).fillna(0.0)
    return (
        history.pivot_table(
            index="date",
            columns="asset_class",
            values="market_value",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
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
