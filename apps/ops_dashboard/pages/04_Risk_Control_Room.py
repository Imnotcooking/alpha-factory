"""Dedicated risk control room for live and paper operations."""

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

from oqp.accounts import (  # noqa: E402
    account_asset_summary,
    account_nav_drawdowns,
    account_performance_summary,
    account_position_history_by_asset,
    account_position_history_by_symbol,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_position_history,
    load_latest_account_nav,
    load_latest_account_positions,
)
from oqp.config import load_settings  # noqa: E402
from oqp.ops import collect_ops_status  # noqa: E402
from oqp.options import option_leg_report, recognize_option_spreads, underlying_exposure_report  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_orders,
    paper_order_notional_today,
)
from oqp.ui import (  # noqa: E402
    apply_ops_theme,
    language_selector,
    ops_tabs,
    ops_text,
    page_header,
    render_dark_line_chart,
    render_dark_table,
    style_dark_plotly,
)


st.set_page_config(
    page_title="Risk Control Room",
    layout="wide",
    page_icon="RISK",
    initial_sidebar_state="expanded",
)

OPS_LANG = language_selector()
apply_ops_theme()


def T(key: str, default: str | None = None, **format_values: Any) -> str:
    return ops_text(OPS_LANG, key, default, **format_values)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def money(value: Any) -> str:
    number = _float(value)
    return "missing" if number is None else f"{number:,.2f}"


def signed_money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:,.2f}"


def percent(value: Any) -> str:
    number = _float(value)
    return "missing" if number is None else f"{number * 100:.2f}%"


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


@st.cache_data(ttl=60)
def load_risk_room_data() -> dict[str, Any]:
    settings = load_settings()
    account_ledger = default_account_ledger_path()
    paper_ledger = default_paper_trading_ledger_path()
    ops_snapshot = collect_ops_status(settings=settings)
    return {
        "settings": settings,
        "account_ledger": account_ledger,
        "paper_ledger": paper_ledger,
        "ops_snapshot": ops_snapshot,
        "live": load_account_bundle(account_ledger, "live"),
        "paper": load_account_bundle(account_ledger, "paper"),
        "paper_orders": load_latest_paper_orders(paper_ledger, limit=150),
        "paper_reviews": load_latest_paper_execution_reviews(paper_ledger, limit=150),
        "paper_daily_notional": paper_order_notional_today(paper_ledger),
    }


def load_account_bundle(account_ledger: Path, environment: str) -> dict[str, Any]:
    latest_nav = load_latest_account_nav(account_ledger, environment=environment)
    nav_raw = load_account_nav_history(account_ledger, environment=environment)
    nav_history = account_nav_drawdowns(nav_raw)
    positions = load_latest_account_positions(account_ledger, environment=environment)
    position_history = load_account_position_history(account_ledger, environment=environment)
    nav = latest_value(latest_nav, "net_liquidation")
    cash = latest_value(latest_nav, "cash")
    daily_pnl = latest_value(latest_nav, "daily_pnl")
    performance = account_performance_summary(
        nav_raw,
        positions,
        current_nav=nav,
        current_cash=cash,
        current_daily_pnl=daily_pnl,
    )
    return {
        "latest_nav": latest_nav,
        "nav_raw": nav_raw,
        "nav_history": nav_history,
        "positions": positions,
        "position_history": position_history,
        "nav": nav,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "performance": performance,
    }


def latest_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    return _float(frame.iloc[0].get(column))


def latest_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "missing"
    value = frame.iloc[0].get(column)
    return "missing" if value in (None, "") else str(value)


def account_summary_rows(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for account in ("live", "paper"):
        bundle = data[account]
        performance = bundle["performance"]
        rows.append(
            {
                "Account": account.title(),
                "NAV": bundle["nav"],
                "Cash": bundle["cash"],
                "Cash %": performance.get("cash_pct"),
                "Daily P&L": bundle["daily_pnl"],
                "Daily Return": performance.get("daily_return"),
                "Gross Exposure": performance.get("gross_exposure"),
                "Gross Exposure / NAV": performance.get("gross_exposure_pct"),
                "Max Drawdown": performance.get("max_drawdown_pct"),
                "Unrealized P&L": performance.get("unrealized_pnl"),
                "Positions": len(bundle["positions"]),
                "Snapshots": len(bundle["nav_raw"]),
                "As Of": latest_text(bundle["latest_nav"], "as_of"),
            }
        )
    return pd.DataFrame(rows)


def combined_positions(data: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for account in ("live", "paper"):
        positions = data[account]["positions"].copy()
        if positions.empty:
            continue
        positions["Account"] = account.title()
        frames.append(positions)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combined_asset_mix(data: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for account in ("live", "paper"):
        mix = account_asset_summary(data[account]["positions"])
        if mix.empty:
            continue
        mix["Account"] = account.title()
        frames.append(mix)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def concentration_frame(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for account in ("live", "paper"):
        positions = data[account]["positions"]
        nav = data[account]["nav"]
        if positions.empty:
            continue
        out = positions.copy()
        for column in ("market_value", "unrealized_pnl", "realized_pnl"):
            if column not in out.columns:
                out[column] = 0.0
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
        grouped = (
            out.groupby(["symbol", "asset_class"], dropna=False)
            .agg(
                market_value=("market_value", "sum"),
                unrealized_pnl=("unrealized_pnl", "sum"),
                realized_pnl=("realized_pnl", "sum"),
            )
            .reset_index()
        )
        grouped["abs_market_value"] = grouped["market_value"].abs()
        grouped["weight"] = 0.0 if nav in (None, 0) else grouped["abs_market_value"] / float(nav)
        grouped["Account"] = account.title()
        rows.extend(grouped.to_dict("records"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Account",
                "Symbol",
                "Asset Class",
                "Market Value",
                "Weight",
                "Unrealized P&L",
                "Realized P&L",
            ]
        )
    return (
        frame.sort_values("abs_market_value", ascending=False)
        .rename(
            columns={
                "symbol": "Symbol",
                "asset_class": "Asset Class",
                "market_value": "Market Value",
                "weight": "Weight",
                "unrealized_pnl": "Unrealized P&L",
                "realized_pnl": "Realized P&L",
            }
        )
        .reindex(
            columns=[
                "Account",
                "Symbol",
                "Asset Class",
                "Market Value",
                "Weight",
                "Unrealized P&L",
                "Realized P&L",
            ]
        )
    )


def risk_alerts(data: dict[str, Any], summary: pd.DataFrame, concentration: pd.DataFrame) -> pd.DataFrame:
    settings = data["settings"]
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        account = str(row["Account"])
        nav = _float(row.get("NAV"))
        if nav is None or nav <= 0:
            rows.append(alert(account, "NAV missing", "warn", "No positive NAV is available."))
        gross_nav = _float(row.get("Gross Exposure / NAV", row.get("Gross / NAV")))
        if gross_nav is not None and settings.max_gross_exposure is not None and gross_nav > settings.max_gross_exposure:
            rows.append(alert(account, "Gross exposure", "warn", f"{gross_nav:.2f} exceeds configured max {settings.max_gross_exposure:.2f}."))
        max_drawdown = _float(row.get("Max Drawdown"))
        if max_drawdown is not None and settings.max_daily_loss_pct is not None and abs(max_drawdown) > settings.max_daily_loss_pct:
            rows.append(alert(account, "Drawdown", "warn", f"{max_drawdown:.2%} exceeds configured daily-loss reference {settings.max_daily_loss_pct:.2%}."))
        cash_pct = _float(row.get("Cash %"))
        if cash_pct is not None and cash_pct < 0.03:
            rows.append(alert(account, "Cash buffer", "warn", f"{cash_pct:.2%} cash is below 3%."))

    if not concentration.empty:
        heavy = concentration[pd.to_numeric(concentration["Weight"], errors="coerce").fillna(0.0) > 0.25]
        for row in heavy.to_dict("records"):
            rows.append(alert(str(row["Account"]), "Concentration", "warn", f"{row['Symbol']} is {row['Weight']:.2%} of NAV."))

    ops_rows = pd.DataFrame(data["ops_snapshot"].item_rows)
    if not ops_rows.empty:
        broken = ops_rows[ops_rows["Status"].astype(str).str.lower().isin(["fail", "warn"])]
        for row in broken.head(10).to_dict("records"):
            rows.append(alert("System", str(row.get("Check")), str(row.get("Status", "warn")), str(row.get("Detail", ""))))

    if settings.allow_live_trading:
        rows.append(alert("Live", "Live trading gate", "block", "ALLOW_LIVE_TRADING is true. Confirm this is intentional."))
    if settings.allow_paper_order_submit:
        rows.append(alert("Paper", "Paper submit gate", "warn", "ALLOW_PAPER_ORDER_SUBMIT is armed."))

    return pd.DataFrame(rows, columns=["Account", "Check", "Severity", "Detail"])


def alert(account: str, check: str, severity: str, detail: str) -> dict[str, str]:
    return {"Account": account, "Check": check, "Severity": severity, "Detail": detail}


def display_table(frame: pd.DataFrame, *, empty: str = "No rows available.") -> None:
    render_dark_table(frame, empty_message=empty, max_height_px=520)


def plot_bar(frame: pd.DataFrame, *, x: str, y: str, color: str | None = None, title: str = "") -> None:
    if frame.empty or x not in frame or y not in frame:
        st.info(f"No data available for {title or y}.")
        return
    traces = []
    if color and color in frame:
        for name, group in frame.groupby(color):
            traces.append(go.Bar(x=group[x], y=group[y], name=str(name)))
    else:
        traces.append(go.Bar(x=frame[x], y=frame[y], name=y))
    fig = go.Figure(data=traces)
    fig.update_layout(height=360, margin=dict(t=35, b=20, l=10, r=10), barmode="group", title=title)
    style_dark_plotly(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def nav_chart(data: dict[str, Any], columns: list[str], title: str) -> None:
    fig = go.Figure()
    for account in ("live", "paper"):
        history = data[account]["nav_history"]
        if history.empty:
            continue
        history = history.copy()
        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        for column in columns:
            if column in history:
                fig.add_scatter(
                    x=history["date"],
                    y=history[column],
                    name=f"{account.title()} {column.replace('_', ' ').title()}",
                    mode="lines",
                )
    if not fig.data:
        st.info(f"No data available for {title}.")
        return
    fig.update_layout(height=360, margin=dict(t=35, b=20, l=10, r=10), title=title, hovermode="x unified")
    style_dark_plotly(fig)
    st.plotly_chart(fig, use_container_width=True, theme=None)


def cumulative_concentration(concentration: pd.DataFrame) -> pd.DataFrame:
    if concentration.empty:
        return pd.DataFrame(columns=["Account", "Rank", "Cumulative Weight"])
    rows = []
    for account, group in concentration.groupby("Account"):
        local = group.copy()
        local["Weight"] = pd.to_numeric(local["Weight"], errors="coerce").fillna(0.0)
        local = local.sort_values("Weight", ascending=False).reset_index(drop=True)
        local["Rank"] = local.index + 1
        local["Cumulative Weight"] = local["Weight"].cumsum()
        rows.extend(local[["Account", "Rank", "Cumulative Weight"]].to_dict("records"))
    return pd.DataFrame(rows)


def stress_scenarios(data: dict[str, Any]) -> pd.DataFrame:
    scenarios = [
        ("Broad market -3%", -0.03, "equity, etf, option"),
        ("Broad market -5%", -0.05, "equity, etf, option"),
        ("Growth shock -8%", -0.08, "equity, etf, option"),
        ("Single-name gap -12%", -0.12, "largest position"),
        ("Relief rally +3%", 0.03, "equity, etf, option"),
    ]
    rows = []
    concentration = concentration_frame(data)
    for account in ("live", "paper"):
        positions = data[account]["positions"].copy()
        if positions.empty:
            continue
        positions["market_value"] = pd.to_numeric(positions.get("market_value", 0.0), errors="coerce").fillna(0.0)
        gross = positions["market_value"].sum()
        largest = 0.0
        if not concentration.empty:
            account_conc = concentration[concentration["Account"].eq(account.title())]
            if not account_conc.empty:
                largest = float(pd.to_numeric(account_conc["Market Value"], errors="coerce").abs().max())
        for name, shock, scope in scenarios:
            impact_base = largest if scope == "largest position" else gross
            rows.append(
                {
                    "Account": account.title(),
                    "Scenario": name,
                    "Shock": shock,
                    "Estimated P&L": impact_base * shock,
                    "NAV Impact": None if data[account]["nav"] in (None, 0) else (impact_base * shock) / float(data[account]["nav"]),
                    "Scope": scope,
                }
            )
    return pd.DataFrame(rows)


def option_frames(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = combined_positions(data)
    if positions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    legs = option_leg_report(positions)
    spreads = recognize_option_spreads(positions)
    underlying = underlying_exposure_report(positions)
    return legs, spreads, underlying


def allocation_frame(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for account in ("live", "paper"):
        bundle = data[account]
        nav = _float(bundle["nav"]) or 0.0
        cash = _float(bundle["cash"]) or 0.0
        if nav > 0:
            rows.append(
                {
                    "Account": account.title(),
                    "Bucket": "Cash",
                    "Symbol": "Cash",
                    "Asset Class": "cash",
                    "Market Value": cash,
                    "Weight": cash / nav,
                    "Allocation Role": "liquidity",
                }
            )
        positions = bundle["positions"]
        if positions.empty:
            continue
        local = positions.copy()
        for column in ("market_value", "unrealized_pnl"):
            if column not in local.columns:
                local[column] = 0.0
            local[column] = pd.to_numeric(local[column], errors="coerce").fillna(0.0)
        grouped = (
            local.groupby(["symbol", "asset_class"], dropna=False)
            .agg(market_value=("market_value", "sum"), unrealized_pnl=("unrealized_pnl", "sum"))
            .reset_index()
        )
        for row in grouped.to_dict("records"):
            market_value = _float(row.get("market_value")) or 0.0
            asset_class = str(row.get("asset_class") or "unknown")
            rows.append(
                {
                    "Account": account.title(),
                    "Bucket": str(row.get("symbol") or "unknown"),
                    "Symbol": str(row.get("symbol") or "unknown"),
                    "Asset Class": asset_class,
                    "Market Value": market_value,
                    "Weight": 0.0 if nav <= 0 else market_value / nav,
                    "Allocation Role": allocation_role(asset_class),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["Account", "Bucket", "Symbol", "Asset Class", "Market Value", "Weight", "Allocation Role"],
    )


def allocation_role(asset_class: str) -> str:
    normalized = asset_class.strip().lower()
    if normalized in {"cash", "money_market"}:
        return "liquidity"
    if normalized in {"equity", "etf"}:
        return "growth / beta"
    if "option" in normalized:
        return "convexity / hedge"
    if "bond" in normalized or "fixed" in normalized:
        return "defensive"
    return "other"


def allocation_summary(allocation: pd.DataFrame) -> pd.DataFrame:
    columns = ["Account", "Allocation Role", "Market Value", "Weight"]
    if allocation.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        allocation.groupby(["Account", "Allocation Role"], dropna=False)
        .agg(market_value=("Market Value", "sum"))
        .reset_index()
    )
    totals = grouped.groupby("Account")["market_value"].transform(lambda value: value.abs().sum())
    grouped["Weight"] = grouped["market_value"] / totals.replace(0, pd.NA)
    return grouped.rename(columns={"market_value": "Market Value"}).reindex(columns=columns)


def policy_rows(data: dict[str, Any], summary: pd.DataFrame) -> pd.DataFrame:
    settings = data["settings"]
    live_gross = _float(summary.loc[summary["Account"].eq("Live"), "Gross Exposure / NAV"].iloc[0]) if not summary.empty and "Gross Exposure / NAV" in summary else None
    paper_daily_notional = data["paper_daily_notional"]
    rows = [
        {
            "Policy": "Allow live trading",
            "Current": yes_no(settings.allow_live_trading),
            "Limit": "must usually be false",
            "Status": "fail" if settings.allow_live_trading else "pass",
        },
        {
            "Policy": "Allow paper trading",
            "Current": yes_no(settings.allow_paper_trading),
            "Limit": "true only when intentionally running paper",
            "Status": "pass" if settings.allow_paper_trading else "warn",
        },
        {
            "Policy": "Allow paper submit",
            "Current": yes_no(settings.allow_paper_order_submit),
            "Limit": "armed only for broker submit windows",
            "Status": "warn" if settings.allow_paper_order_submit else "pass",
        },
        {
            "Policy": "Max gross exposure",
            "Current": percent(live_gross),
            "Limit": "unset" if settings.max_gross_exposure is None else percent(settings.max_gross_exposure),
            "Status": "pass" if settings.max_gross_exposure is None or live_gross is None or live_gross <= settings.max_gross_exposure else "warn",
        },
        {
            "Policy": "Paper daily notional",
            "Current": money(paper_daily_notional),
            "Limit": money(settings.paper_max_daily_notional),
            "Status": "pass" if settings.paper_max_daily_notional is None or paper_daily_notional <= settings.paper_max_daily_notional else "warn",
        },
        {
            "Policy": "Paper max order notional",
            "Current": "configured",
            "Limit": money(settings.paper_max_order_notional),
            "Status": "pass",
        },
        {
            "Policy": "Paper options enabled",
            "Current": yes_no(settings.paper_options_enabled),
            "Limit": "enabled only when option policy is ready",
            "Status": "pass" if not settings.paper_options_enabled else "warn",
        },
    ]
    return pd.DataFrame(rows)


def progress_metric(label: str, value: float | None, limit: float | None, *, inverse: bool = False) -> None:
    st.caption(label)
    if value is None or limit in (None, 0):
        st.progress(0.0, text="not configured")
        return
    ratio = abs(value) / abs(limit)
    status = "ok" if (ratio <= 1 if not inverse else ratio >= 1) else "over"
    st.progress(min(float(ratio), 1.0), text=f"{ratio:.0%} of limit ({status})")


data = load_risk_room_data()
settings = data["settings"]
summary = account_summary_rows(data)
concentration = concentration_frame(data)
alerts = risk_alerts(data, summary, concentration)
asset_mix = combined_asset_mix(data)
stress = stress_scenarios(data)
legs, spreads, underlying = option_frames(data)
allocation = allocation_frame(data)
allocation_roles = allocation_summary(allocation)


page_header(
    title="Risk Control Room",
    title_zh="风险控制室",
    subtitle="Daily risk cockpit for live and paper accounts: exposure, concentration, drawdown, options, stress tests, and policy gates.",
    subtitle_zh="实盘与模拟账户的日常风险驾驶舱：敞口、集中度、回撤、期权、压力测试与政策闸门。",
    language=OPS_LANG,
)

latest_live = data["live"]
latest_paper = data["paper"]
top = st.columns(8)
top[0].metric(T("live_nav"), money(latest_live["nav"]))
top[1].metric(T("paper_nav"), money(latest_paper["nav"]))
top[2].metric(T("live_daily_pnl", "Live Daily P&L"), signed_money(latest_live["daily_pnl"]))
top[3].metric(T("paper_daily_pnl", "Paper Daily P&L"), signed_money(latest_paper["daily_pnl"]))
top[4].metric(T("live_drawdown", "Live Drawdown"), percent(latest_live["performance"].get("max_drawdown_pct")))
top[5].metric(T("paper_drawdown", "Paper Drawdown"), percent(latest_paper["performance"].get("max_drawdown_pct")))
top[6].metric(T("live_gross_nav", "Live Gross Exposure / NAV"), percent(latest_live["performance"].get("gross_exposure_pct")))
top[7].metric(T("risk_flags", "Risk Flags"), str(len(alerts)))

if alerts.empty:
    st.success("No risk flags from current page checks.")
elif alerts["Severity"].astype(str).str.lower().eq("block").any():
    st.error("At least one blocking risk flag needs attention.")
else:
    st.warning(f"{len(alerts)} risk flag(s) need review.")

overview_tab, exposure_tab, drawdown_tab, options_tab, stress_tab, allocation_tab = st.tabs(
    ops_tabs(OPS_LANG, "risk_tabs")
)

with overview_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Account Summary")
        display_table(summary, empty="No live or paper account summary is available.")
    with right:
        st.subheader("Risk Alerts")
        display_table(alerts, empty="No risk alerts.")

    st.subheader("Operating Snapshot")
    status_rows = pd.DataFrame(data["ops_snapshot"].item_rows)
    if status_rows.empty:
        st.info("No ops status rows are available.")
    else:
        display_table(status_rows.head(20))

with exposure_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Asset Class Exposure")
        plot_bar(asset_mix, x="Asset Class", y="Market Value", color="Account", title="Market Value by Asset Class")
    with right:
        st.subheader("Account Exposure")
        exposure = summary.reindex(columns=["Account", "NAV", "Cash", "Gross Exposure"])
        display_table(exposure)
        if not exposure.empty:
            long = exposure.melt(id_vars=["Account"], value_vars=["NAV", "Cash", "Gross Exposure"], var_name="Metric", value_name="Value")
            plot_bar(long, x="Metric", y="Value", color="Account", title="NAV, Cash, Gross Exposure")

    st.subheader("Historical Asset Exposure")
    for account in ("live", "paper"):
        history = account_position_history_by_asset(data[account]["position_history"])
        if history.empty:
            st.info(f"No {account} asset exposure history yet.")
            continue
        pivot = history.pivot_table(index="date", columns="asset_class", values="market_value", aggfunc="sum").fillna(0.0)
        render_dark_line_chart(pivot, yaxis_title="Market Value")

    st.subheader("Concentration")
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Top Position Concentration")
        display_table(concentration.head(25), empty="No concentration rows are available.")
    with right:
        st.subheader("Top Weights")
        plot_bar(concentration.head(12), x="Symbol", y="Weight", color="Account", title="Top Position Weights")

    curve = cumulative_concentration(concentration)
    st.subheader("Cumulative Concentration Curve")
    if curve.empty:
        st.info("No concentration curve is available.")
    else:
        fig = go.Figure()
        for account, group in curve.groupby("Account"):
            fig.add_scatter(x=group["Rank"], y=group["Cumulative Weight"], mode="lines+markers", name=str(account))
        fig.update_layout(height=340, margin=dict(t=35, b=20, l=10, r=10), yaxis_tickformat=".0%")
        style_dark_plotly(fig)
        st.plotly_chart(fig, use_container_width=True, theme=None)

with drawdown_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("NAV Equity Curve")
        nav_chart(data, ["net_liquidation"], "Live vs Paper NAV")
    with right:
        st.subheader("Drawdown")
        nav_chart(data, ["drawdown_pct"], "Live vs Paper Drawdown")

    st.subheader("Daily P&L")
    nav_chart(data, ["daily_pnl"], "Live vs Paper Daily P&L")

    st.subheader("Position Exposure History")
    for account in ("live", "paper"):
        history = account_position_history_by_symbol(data[account]["position_history"])
        if history.empty:
            st.info(f"No {account} symbol exposure history yet.")
            continue
        latest_symbols = (
            history.sort_values("date")
            .groupby("symbol")
            .tail(1)
            .assign(abs_value=lambda frame: frame["market_value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(8)["symbol"]
            .tolist()
        )
        chart = history[history["symbol"].isin(latest_symbols)]
        if not chart.empty:
            render_dark_line_chart(
                chart.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").fillna(0.0),
                yaxis_title="Market Value",
            )

with options_tab:
    spread_left, greek_right = st.columns([1.15, 1])
    with spread_left:
        st.subheader("Recognized Option Spreads")
        display_table(spreads, empty="No option spreads are recognized.")
    with greek_right:
        st.subheader("Underlying Option Exposure")
        display_table(underlying, empty="No underlying option exposure rows are available.")

    st.subheader("Option Leg Audit")
    display_table(legs, empty="No option legs are available.")

    if not underlying.empty:
        greek_cols = [column for column in ["Net Option Delta", "Option Market Value"] if column in underlying.columns]
        if greek_cols:
            chart = underlying[["Underlying", *greek_cols]].copy()
            long = chart.melt(id_vars=["Underlying"], var_name="Metric", value_name="Value")
            plot_bar(long, x="Underlying", y="Value", color="Metric", title="Option Risk by Underlying")

with stress_tab:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Scenario Impact")
        display_table(stress, empty="No stress scenarios are available.")
    with right:
        st.subheader("Scenario P&L")
        plot_bar(stress, x="Scenario", y="Estimated P&L", color="Account", title="Estimated Stress P&L")

    if not stress.empty:
        worst = stress.sort_values("Estimated P&L").head(5)
        st.subheader("Worst Reads")
        display_table(worst)

with allocation_tab:
    st.subheader("Current Allocation")
    left, right = st.columns([1.1, 1])
    with left:
        display_table(allocation_roles, empty="No allocation role rows are available.")
    with right:
        plot_bar(allocation_roles, x="Allocation Role", y="Weight", color="Account", title="Allocation Role Weights")

    st.subheader("Symbol Weights")
    symbol_weights = allocation.copy()
    if not symbol_weights.empty:
        symbol_weights["Abs Weight"] = pd.to_numeric(symbol_weights["Weight"], errors="coerce").abs()
        symbol_weights = symbol_weights.sort_values(["Account", "Abs Weight"], ascending=[True, False])
    display_table(
        symbol_weights.drop(columns=["Abs Weight"], errors="ignore").head(40),
        empty="No symbol allocation rows are available.",
    )

    st.subheader("Target Weight Sketch")
    account_choice = st.selectbox(T("account", "Account"), ["Live", "Paper"], key="risk_allocation_account")
    account_nav = data[account_choice.lower()]["nav"] or 0.0
    current = allocation_roles[allocation_roles["Account"].eq(account_choice)].copy()
    current_map = {
        str(row["Allocation Role"]): _float(row.get("Weight")) or 0.0
        for row in current.to_dict("records")
    }
    target = pd.DataFrame(
        [
            {"Allocation Role": "liquidity", "Target Weight": 0.05},
            {"Allocation Role": "growth / beta", "Target Weight": 0.75},
            {"Allocation Role": "convexity / hedge", "Target Weight": 0.10},
            {"Allocation Role": "defensive", "Target Weight": 0.10},
            {"Allocation Role": "other", "Target Weight": 0.00},
        ]
    )
    edited = st.data_editor(
        target,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Target Weight": st.column_config.NumberColumn(
                "Target Weight",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f",
            )
        },
        key=f"risk_allocation_targets_{account_choice}",
    )
    gap = edited.copy()
    gap["Current Weight"] = gap["Allocation Role"].map(current_map).fillna(0.0)
    gap["Gap"] = gap["Target Weight"] - gap["Current Weight"]
    gap["Implied Move"] = gap["Gap"] * float(account_nav or 0.0)
    display_table(gap)
    if not gap.empty:
        plot_bar(gap, x="Allocation Role", y="Gap", title=f"{account_choice} Target Allocation Gap")

    st.info("This tab is currently diagnostic only. HRP, Kelly sizing, regime-aware allocation, and rebalance proposals can plug into this surface later.")
