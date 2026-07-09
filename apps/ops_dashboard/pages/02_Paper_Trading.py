"""Paper Trading page for the unified Ops dashboard."""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    account_asset_exposure_pivot,
    account_asset_summary,
    account_nav_drawdowns,
    account_performance_summary,
    account_position_history_by_asset,
    account_position_history_by_symbol,
    account_trade_event_summary,
    account_trade_events_display,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_position_history,
    load_account_trade_events,
    load_latest_account_nav,
    load_latest_account_positions,
)
from oqp.config import load_settings  # noqa: E402
from oqp.ops import collect_ops_status  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    PaperOrderTicketStatus,
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_nav,
    load_latest_paper_orders,
    load_latest_paper_positions,
    load_paper_strategy_registry,
    paper_order_notional_today,
)
from oqp.ui import (  # noqa: E402
    apply_ops_theme,
    language_selector,
    ops_tabs,
    ops_text,
    page_header,
    qmt_connector_contract_frame,
    qmt_safety_gate_frame,
    qmt_strategy_route_frame,
    render_dark_area_chart,
    render_dark_bar_chart,
    render_dark_line_chart,
    render_dark_table,
    render_market_lane_chips,
    render_qmt_account_panel,
    render_qmt_audit_panel,
    render_qmt_connector_panel,
    render_qmt_safety_panel,
)


st.set_page_config(
    page_title="Paper Trading",
    layout="wide",
    page_icon="PAPER",
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


def compact_money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if absolute >= 100_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,.2f}"


def signed_money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:,.2f}"


def signed_compact_money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{sign}{number / 1_000_000:.2f}M"
    if absolute >= 100_000:
        return f"{sign}{number / 1_000:.1f}K"
    return f"{sign}{number:,.2f}"


def percent(value: Any) -> str:
    number = _float(value)
    return "missing" if number is None else f"{number * 100:.2f}%"


def quantity(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    if abs(number - round(number)) < 1e-8:
        return f"{number:,.0f}"
    return f"{number:,.2f}"


def human_timestamp(value: Any) -> str:
    if value in (None, ""):
        return "missing"
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %d %H:%M UTC")


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


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]


def render_metric_strip(items: list[dict[str, str]]) -> None:
    cards = []
    for item in items:
        label = escape(str(item.get("label") or ""))
        value = escape(str(item.get("value") or "missing"))
        detail = escape(str(item.get("detail") or ""))
        tone = escape(str(item.get("tone") or "neutral"))
        title = escape(str(item.get("title") or item.get("detail") or item.get("value") or ""))
        cards.append(
            dedent(
                f"""
            <div class="oqp-paper-kpi oqp-paper-kpi-{tone}" title="{title}">
                <span>{label}</span>
                <strong>{value}</strong>
                <small>{detail}</small>
            </div>
            """
            ).strip()
        )
    st.markdown(
        dedent(
            f"""
<style>
.oqp-paper-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin: 0.9rem 0 1.25rem 0;
}}
.oqp-paper-kpi {{
    min-height: 5.35rem;
    border-radius: 10px;
    padding: 0.78rem 0.88rem;
    background:
        radial-gradient(circle at 18% 0%, rgba(45, 212, 191, 0.10), transparent 34%),
        linear-gradient(180deg, rgba(17, 28, 43, 0.82), rgba(8, 13, 21, 0.72));
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.025),
        0 12px 24px rgba(0, 0, 0, 0.14);
    overflow: hidden;
}}
.oqp-paper-kpi span {{
    display: block;
    color: #9BAAC0;
    font-size: 0.74rem;
    font-weight: 760;
    letter-spacing: 0.045em;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.oqp-paper-kpi strong {{
    display: block;
    color: #F8FAFC;
    font-size: clamp(1.18rem, 1.8vw, 1.72rem);
    line-height: 1.0;
    margin-top: 0.32rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.oqp-paper-kpi small {{
    display: block;
    min-height: 1rem;
    margin-top: 0.38rem;
    color: #7F8FA7;
    font-size: 0.72rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.oqp-paper-kpi-good {{ box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 0 0 1px rgba(45,212,191,0.09), 0 12px 24px rgba(0,0,0,0.14); }}
.oqp-paper-kpi-warn {{ box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 0 0 1px rgba(245,158,11,0.12), 0 12px 24px rgba(0,0,0,0.14); }}
.oqp-paper-kpi-locked {{ box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 0 0 1px rgba(96,165,250,0.10), 0 12px 24px rgba(0,0,0,0.14); }}
</style>
<div class="oqp-paper-kpi-grid">{''.join(cards)}</div>
"""
        ).strip(),
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60)
def load_paper_room_data() -> dict[str, Any]:
    settings = load_settings()
    account_ledger = default_account_ledger_path()
    paper_ledger = default_paper_trading_ledger_path()
    ops_snapshot = collect_ops_status(settings=settings)
    nav_latest = load_latest_account_nav(account_ledger, environment="paper")
    nav_raw = load_account_nav_history(account_ledger, environment="paper")
    positions = load_latest_account_positions(account_ledger, environment="paper")
    position_history = load_account_position_history(account_ledger, environment="paper")
    events = load_account_trade_events(account_ledger, environment="paper", limit=150)
    legacy_nav = load_latest_paper_nav(paper_ledger)
    legacy_positions = load_latest_paper_positions(paper_ledger)
    orders = load_latest_paper_orders(paper_ledger, limit=150)
    reviews = load_latest_paper_execution_reviews(paper_ledger, limit=150)
    registry = load_paper_strategy_registry(paper_ledger, limit=250)
    daily_notional_used = paper_order_notional_today(paper_ledger)
    return {
        "settings": settings,
        "account_ledger": account_ledger,
        "paper_ledger": paper_ledger,
        "ops_snapshot": ops_snapshot,
        "nav_latest": nav_latest,
        "nav_raw": nav_raw,
        "positions": positions,
        "position_history": position_history,
        "events": events,
        "legacy_nav": legacy_nav,
        "legacy_positions": legacy_positions,
        "orders": orders,
        "reviews": reviews,
        "registry": registry,
        "daily_notional_used": daily_notional_used,
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


def paper_summary(
    nav_raw: pd.DataFrame,
    positions: pd.DataFrame,
    latest_nav: pd.DataFrame,
    legacy_nav: pd.DataFrame,
) -> dict[str, Any]:
    nav = latest_value(latest_nav, "net_liquidation")
    cash = latest_value(latest_nav, "cash")
    daily_pnl = latest_value(latest_nav, "daily_pnl")
    source = "unified account ledger"
    as_of = latest_text(latest_nav, "as_of")
    if nav is None and not legacy_nav.empty:
        nav = latest_value(legacy_nav, "net_liquidation")
        cash = latest_value(legacy_nav, "cash")
        daily_pnl = latest_value(legacy_nav, "daily_pnl")
        source = "paper trading ledger"
        as_of = latest_text(legacy_nav, "as_of")

    performance = account_performance_summary(
        nav_raw,
        positions,
        current_nav=nav,
        current_cash=cash,
        current_daily_pnl=daily_pnl,
    )
    return {
        "nav": nav,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "as_of": human_timestamp(as_of),
        "source": source,
        "positions": len(positions),
        "performance": performance,
    }


def history_dates(frame: pd.DataFrame, column: str = "date") -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(pd.to_datetime(frame[column], errors="coerce").dropna().nunique())


def paper_positions_display(positions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Symbol",
        "Asset Class",
        "Quantity",
        "Avg Cost",
        "Cost Basis",
        "Market Price",
        "Market Value",
        "Unrealized P&L",
        "Currency",
        "As Of",
    ]
    if positions.empty:
        return pd.DataFrame(columns=columns)

    out = positions.copy()
    for column in ("quantity", "average_cost", "market_price", "market_value", "unrealized_pnl", "multiplier"):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["multiplier"] = out["multiplier"].fillna(1.0)
    out["cost_basis"] = out["quantity"] * out["average_cost"] * out["multiplier"]
    display = (
        out.reindex(
            columns=[
                "symbol",
                "asset_class",
                "quantity",
                "average_cost",
                "cost_basis",
                "market_price",
                "market_value",
                "unrealized_pnl",
                "currency",
                "as_of",
            ]
        )
        .rename(
            columns={
                "symbol": "Symbol",
                "asset_class": "Asset Class",
                "quantity": "Quantity",
                "average_cost": "Avg Cost",
                "cost_basis": "Cost Basis",
                "market_price": "Market Price",
                "market_value": "Market Value",
                "unrealized_pnl": "Unrealized P&L",
                "currency": "Currency",
                "as_of": "As Of",
            }
        )
        .reindex(columns=columns)
    )
    for column in ("Avg Cost", "Cost Basis", "Market Price", "Market Value"):
        if column in display:
            display[column] = display[column].map(money)
    if "Unrealized P&L" in display:
        display["Unrealized P&L"] = display["Unrealized P&L"].map(signed_money)
    if "Quantity" in display:
        display["Quantity"] = display["Quantity"].map(quantity)
    if "As Of" in display:
        display["As Of"] = display["As Of"].map(human_timestamp)
    return display


def ticket_status_frame(orders: pd.DataFrame) -> pd.DataFrame:
    columns = ["Status", "Tickets", "Meaning"]
    if orders.empty or "status" not in orders.columns:
        return pd.DataFrame(columns=columns)
    counts = orders["status"].astype(str).value_counts().to_dict()
    rows = [
        {
            "Status": PaperOrderTicketStatus.DRY_RUN.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.DRY_RUN.value, 0)),
            "Meaning": "created by paper runner; needs human decision",
        },
        {
            "Status": PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value, 0)),
            "Meaning": "approved by human; still blocked unless submit gate is armed",
        },
        {
            "Status": PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value, 0)),
            "Meaning": "sent to IBKR paper account",
        },
        {
            "Status": PaperOrderTicketStatus.REJECTED.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.REJECTED.value, 0)),
            "Meaning": "closed by human rejection",
        },
    ]
    known = {row["Status"] for row in rows}
    other_count = sum(int(value) for key, value in counts.items() if key not in known)
    if other_count:
        rows.append({"Status": "other", "Tickets": other_count, "Meaning": "unexpected status"})
    return pd.DataFrame(rows, columns=columns)


def review_status_frame(reviews: pd.DataFrame) -> pd.DataFrame:
    columns = ["Decision", "Reviews", "Estimated Notional"]
    if reviews.empty or "decision" not in reviews.columns:
        return pd.DataFrame(columns=columns)
    out = reviews.copy()
    out["estimated_notional"] = pd.to_numeric(out["estimated_notional"], errors="coerce").fillna(0.0)
    return (
        out.groupby("decision")
        .agg(reviews=("proposal_id", "count"), estimated_notional=("estimated_notional", "sum"))
        .reset_index()
        .rename(columns={"decision": "Decision", "reviews": "Reviews", "estimated_notional": "Estimated Notional"})
        .reindex(columns=columns)
    )


def review_status_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "Estimated Notional" in out:
        out["Estimated Notional"] = out["Estimated Notional"].map(money)
    if "Reviews" in out:
        out["Reviews"] = out["Reviews"].map(quantity)
    return out


def asset_mix_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Market Value", "Unrealized P&L"):
        if column in out:
            formatter = signed_money if "P&L" in column else money
            out[column] = out[column].map(formatter)
    if "Rows" in out:
        out["Rows"] = out["Rows"].map(quantity)
    return out


def nav_snapshot_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("net_liquidation", "cash", "daily_pnl", "equity_peak"):
        if column in out:
            formatter = signed_money if column == "daily_pnl" else money
            out[column] = out[column].map(formatter)
    for column in ("drawdown_pct", "cumulative_return"):
        if column in out:
            out[column] = out[column].map(percent)
    if "position_count" in out:
        out["position_count"] = out["position_count"].map(quantity)
    for column in ("date", "as_of"):
        if column in out:
            out[column] = out[column].map(human_timestamp)
    return out


def ticket_display(orders: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Created",
        "Status",
        "Symbol",
        "Side",
        "Quantity",
        "Type",
        "Limit",
        "Strategy",
        "Proposal",
        "Broker Submit",
        "Ticket",
    ]
    if orders.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in orders.to_dict("records"):
        metadata = parse_metadata(row.get("metadata_json"))
        rows.append(
            {
                "Created": human_timestamp(row.get("created_at")),
                "Status": row.get("status"),
                "Symbol": row.get("symbol"),
                "Side": row.get("side"),
                "Quantity": quantity(row.get("quantity")),
                "Type": row.get("order_type"),
                "Limit": money(row.get("limit_price")),
                "Strategy": row.get("strategy_id") or "",
                "Proposal": metadata.get("proposal_id", ""),
                "Broker Submit": yes_no(bool(metadata.get("broker_submit_enabled"))),
                "Ticket": row.get("order_id"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def review_display(reviews: pd.DataFrame) -> pd.DataFrame:
    columns = ["Reviewed", "Decision", "Proposal", "Orders", "Estimated Notional", "Message"]
    if reviews.empty:
        return pd.DataFrame(columns=columns)
    out = reviews.copy()
    out["Reviewed"] = out["reviewed_at"].map(human_timestamp)
    out["Estimated Notional"] = out["estimated_notional"].map(money)
    out["Orders"] = out["order_count"].map(quantity)
    return (
        out.rename(
            columns={
                "decision": "Decision",
                "proposal_id": "Proposal",
                "message": "Message",
            }
        )
        .reindex(columns=columns)
    )


def registry_display(registry: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Market",
        "Candidate",
        "Status",
        "Kill Switch",
        "Max Order",
        "Max Daily",
        "Symbols",
        "Approved",
        "Notes",
    ]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in registry.to_dict("records"):
        rows.append(
            {
                "Strategy": row.get("strategy_id"),
                "Market": row.get("market_vertical"),
                "Candidate": row.get("candidate_id"),
                "Status": row.get("status"),
                "Kill Switch": yes_no(bool(row.get("kill_switch"))),
                "Max Order": money(row.get("max_order_notional")),
                "Max Daily": money(row.get("max_daily_notional")),
                "Symbols": ", ".join(parse_json_list(row.get("allowed_symbols_json"))),
                "Approved": human_timestamp(row.get("approved_at")),
                "Notes": row.get("notes"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def paper_health_rows(snapshot: Any, settings: Any) -> pd.DataFrame:
    frame = pd.DataFrame(snapshot.item_rows)
    if frame.empty:
        return pd.DataFrame(columns=["Category", "Check", "Status", "Detail"])
    mask = (
        frame["Check"].str.contains("paper", case=False, na=False)
        | frame["Category"].isin(["Safety", "Gateway", "Broker Heartbeat", "Jobs", "Schedulers"])
    )
    rows = frame[mask].copy()
    control_rows = pd.DataFrame(
        [
            {
                "Category": "Safety",
                "Check": "ALLOW_LIVE_TRADING",
                "Status": "pass" if not settings.allow_live_trading else "fail",
                "Detail": f"{str(settings.allow_live_trading).lower()}",
            },
            {
                "Category": "Safety",
                "Check": "ALLOW_PAPER_TRADING",
                "Status": "pass" if settings.allow_paper_trading else "warn",
                "Detail": f"{str(settings.allow_paper_trading).lower()}",
            },
            {
                "Category": "Safety",
                "Check": "ALLOW_PAPER_ORDER_SUBMIT",
                "Status": "warn" if settings.allow_paper_order_submit else "pass",
                "Detail": f"{str(settings.allow_paper_order_submit).lower()}",
            },
        ]
    )
    return pd.concat([control_rows, rows], ignore_index=True).reindex(
        columns=["Category", "Check", "Status", "Detail"]
    )


data = load_paper_room_data()
settings = data["settings"]
latest_nav = data["nav_latest"]
nav_raw = data["nav_raw"]
nav_history = account_nav_drawdowns(nav_raw)
positions = data["positions"]
position_history = data["position_history"]
events = data["events"]
legacy_nav = data["legacy_nav"]
legacy_positions = data["legacy_positions"]
orders = data["orders"]
reviews = data["reviews"]
registry = data["registry"]
summary = paper_summary(nav_raw, positions, latest_nav, legacy_nav)
ticket_status = ticket_status_frame(orders)
review_status = review_status_frame(reviews)
asset_mix = account_asset_summary(positions)
asset_history = account_position_history_by_asset(position_history)
symbol_history = account_position_history_by_symbol(position_history)
health = paper_health_rows(data["ops_snapshot"], settings)
paper_review_ready = bool(
    settings.trading_mode.lower() == "paper"
    and not settings.allow_live_trading
    and settings.allow_paper_trading
)


page_header(
    title="Paper Trading",
    title_zh="模拟交易",
    subtitle="Simulation control room for paper account health, strategy authorization, safety reviews, and order tickets.",
    subtitle_zh="模拟账户健康、策略授权、安全审查与订单票据的控制室。",
    language=OPS_LANG,
)
render_market_lane_chips(
    language=OPS_LANG,
    lanes=("EQUITY_US", "OPTIONS_US", "EQUITY_CN", "OPTIONS_CN", "FUTURES_CN"),
    caption="Paper trading currently routes through IBKR for US lanes; CN lanes are staged for 华源证券/QMT once adapters and guardrails are wired.",
)

render_metric_strip(
    [
        {
            "label": T("nav"),
            "value": compact_money(summary["nav"]),
            "detail": money(summary["nav"]),
            "tone": "good",
        },
        {
            "label": T("cash"),
            "value": compact_money(summary["cash"]),
            "detail": money(summary["cash"]),
            "tone": "good",
        },
        {
            "label": T("daily_pnl"),
            "value": signed_compact_money(summary["daily_pnl"]),
            "detail": signed_money(summary["daily_pnl"]),
            "tone": "good" if (_float(summary["daily_pnl"]) or 0.0) >= 0 else "warn",
        },
        {
            "label": T("positions"),
            "value": str(summary["positions"]),
            "detail": f"as of {summary['as_of']}",
        },
        {
            "label": T("unrealized_pnl"),
            "value": signed_compact_money(summary["performance"].get("unrealized_pnl")),
            "detail": signed_money(summary["performance"].get("unrealized_pnl")),
            "tone": "good" if (_float(summary["performance"].get("unrealized_pnl")) or 0.0) >= 0 else "warn",
        },
        {
            "label": T("review_gate"),
            "value": T("open") if paper_review_ready else T("locked"),
            "detail": T("proposal_reviews"),
            "tone": "good" if paper_review_ready else "locked",
        },
        {
            "label": T("submit_gate"),
            "value": T("armed") if settings.allow_paper_order_submit else T("locked"),
            "detail": T("broker_submit"),
            "tone": "warn" if settings.allow_paper_order_submit else "locked",
        },
        {
            "label": T("daily_notional"),
            "value": compact_money(data["daily_notional_used"]),
            "detail": money(data["daily_notional_used"]),
        },
    ]
)

if settings.allow_paper_order_submit:
    st.warning(T("paper_submission_armed_readonly"))
else:
    st.info(T("paper_submit_locked"))

overview_tab, account_tab, strategy_tab, reviews_tab, tickets_tab, ledger_tab, health_tab = st.tabs(
    ops_tabs(OPS_LANG, "paper_tabs")
)

with overview_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Paper NAV")
        if nav_history.empty:
            st.info("No paper NAV history has been recorded yet.")
        elif history_dates(nav_history) < 2:
            st.info("Only one paper NAV snapshot is available. The equity chart will become useful after the next daily snapshot.")
            render_dark_table(
                nav_snapshot_display(
                    nav_history.tail(1).reindex(columns=["date", "net_liquidation", "cash", "daily_pnl", "position_count"])
                ),
            )
        else:
            render_dark_line_chart(
                nav_history.set_index("date")[["net_liquidation", "equity_peak"]],
                yaxis_title="Account Value",
            )
    with right:
        st.subheader("Ticket Queue")
        if ticket_status.empty:
            st.info("No paper order tickets have been recorded yet.")
        else:
            render_dark_table(ticket_status, max_height_px=280)
            render_dark_bar_chart(ticket_status.set_index("Status")[["Tickets"]])

    workflow = pd.DataFrame(
        [
            {
                "Layer": "Approved Strategy",
                "Purpose": "A research candidate is allowed to run in paper.",
                "Stored In": "paper_strategy_registry",
                "Current Signal": f"{len(registry)} registered row(s)",
            },
            {
                "Layer": "Proposal Review",
                "Purpose": "Safety policy checks the proposed orders.",
                "Stored In": "paper_execution_reviews",
                "Current Signal": f"{len(reviews)} recent review row(s)",
            },
            {
                "Layer": "Dry-Run Ticket",
                "Purpose": "A broker-shaped ticket is created for human review.",
                "Stored In": "paper_orders",
                "Current Signal": f"{len(orders)} recent ticket row(s)",
            },
            {
                "Layer": "Submit Gate",
                "Purpose": "Only approved tickets can be sent, and only if submit is armed.",
                "Stored In": "settings + paper order metadata",
                "Current Signal": "armed" if settings.allow_paper_order_submit else "locked",
            },
        ]
    )
    st.subheader("Paper Trading Flow")
    render_dark_table(workflow)

    st.subheader("QMT Paper Connector")
    render_qmt_connector_panel(settings, data["ops_snapshot"], compact=True)

with account_tab:
    chart_left, chart_right = st.columns([1.25, 1])
    with chart_left:
        st.subheader("Equity And Drawdown")
        if nav_history.empty:
            st.info("No NAV history has been recorded yet.")
        elif history_dates(nav_history) < 2:
            st.info("Drawdown needs at least two NAV snapshots. Current snapshot is shown below.")
            render_dark_table(
                nav_snapshot_display(
                    nav_history.tail(1).reindex(columns=["date", "net_liquidation", "cash", "daily_pnl", "drawdown_pct"])
                ),
            )
        else:
            render_dark_line_chart(
                nav_history.set_index("date")[["net_liquidation", "equity_peak"]],
                yaxis_title="Account Value",
            )
            render_dark_line_chart(
                nav_history.set_index("date")[["drawdown_pct"]],
                yaxis_title="Drawdown",
            )
    with chart_right:
        st.subheader("Account State")
        state = pd.DataFrame(
            [
                {"Metric": "Source", "Value": summary["source"]},
                {"Metric": "Snapshot As Of", "Value": summary["as_of"]},
                {"Metric": "Daily Return", "Value": percent(summary["performance"].get("daily_return"))},
                {"Metric": "Cumulative Return", "Value": percent(summary["performance"].get("cumulative_return"))},
                {"Metric": "Cash Weight", "Value": percent(summary["performance"].get("cash_pct"))},
                {"Metric": "Gross Exposure / NAV", "Value": percent(summary["performance"].get("gross_exposure_pct"))},
                {"Metric": "NAV Observations", "Value": str(summary["performance"].get("nav_observations"))},
            ]
        )
        render_dark_table(state)

    holdings_left, holdings_right = st.columns([1.25, 1])
    with holdings_left:
        st.subheader("Current Holdings")
        display = paper_positions_display(positions)
        if display.empty:
            st.info("No unified paper account positions have been recorded yet.")
        else:
            render_dark_table(display, max_height_px=480)
    with holdings_right:
        st.subheader("Asset Mix")
        if asset_mix.empty:
            st.info("No paper asset mix is available.")
        else:
            render_dark_table(asset_mix_display(asset_mix), max_height_px=320)
            render_dark_bar_chart(
                asset_mix.set_index("Asset Class")[["Market Value"]],
                yaxis_title="Market Value",
            )

    st.subheader("QMT Paper Account Slice")
    render_qmt_account_panel(data["ops_snapshot"], positions, environment="paper")

    st.subheader("Exposure History")
    asset_chart = account_asset_exposure_pivot(asset_history)
    if asset_chart.empty:
        st.info("No paper position history has been recorded yet.")
    elif history_dates(asset_history) < 2:
        st.info("Only one paper position snapshot is available. Showing current asset mix until history builds up.")
        if not asset_mix.empty:
            render_dark_bar_chart(
                asset_mix.set_index("Asset Class")[["Market Value"]],
                yaxis_title="Market Value",
            )
    else:
        render_dark_area_chart(asset_chart, yaxis_title="Market Value")

with strategy_tab:
    st.subheader("Paper Strategy Registry")
    display = registry_display(registry)
    if display.empty:
        st.info("No strategy has been registered for paper-running yet.")
    else:
        status_counts = registry["status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Strategies")
        status_left, status_right = st.columns([1, 2])
        with status_left:
            render_dark_table(status_counts, max_height_px=280)
        with status_right:
            render_dark_table(display, max_height_px=420)

    st.subheader("QMT Strategy Route Eligibility")
    render_dark_table(
        qmt_strategy_route_frame(registry),
        empty_message="No paper strategies are registered for QMT route review yet.",
        max_height_px=320,
    )

with reviews_tab:
    status_left, latest_right = st.columns([1, 2])
    with status_left:
        st.subheader("Review Decisions")
        if review_status.empty:
            st.info("No paper execution reviews have been recorded yet.")
        else:
            render_dark_table(review_status_display(review_status), max_height_px=280)
            render_dark_bar_chart(review_status.set_index("Decision")[["Reviews"]])
    with latest_right:
        st.subheader("Latest Safety Reviews")
        display = review_display(reviews)
        if display.empty:
            st.info("No recent paper safety reviews are available.")
        else:
            render_dark_table(display, max_height_px=460)

    st.subheader("QMT Review Gates")
    render_dark_table(qmt_safety_gate_frame(settings), max_height_px=300)

with tickets_tab:
    status_left, latest_right = st.columns([1, 2])
    with status_left:
        st.subheader("Ticket Status")
        if ticket_status.empty:
            st.info("No paper tickets are available.")
        else:
            render_dark_table(ticket_status, max_height_px=280)
            render_dark_bar_chart(ticket_status.set_index("Status")[["Tickets"]])
    with latest_right:
        st.subheader("Latest Tickets")
        display = ticket_display(orders)
        if display.empty:
            st.info("No paper order tickets have been recorded yet.")
        else:
            render_dark_table(display, max_height_px=460)

    st.subheader("QMT Ticket Route State")
    render_dark_table(
        qmt_strategy_route_frame(registry),
        empty_message="No QMT route metadata is available for current paper tickets.",
        max_height_px=260,
    )

    st.subheader("Paper Events")
    if events.empty:
        st.info("No paper account events have been recorded yet.")
    else:
        event_left, event_right = st.columns([2, 1])
        with event_left:
            render_dark_table(account_trade_events_display(events), max_height_px=420)
        with event_right:
            render_dark_table(account_trade_event_summary(events), max_height_px=420)

with ledger_tab:
    ledger_cols = st.columns(5)
    ledger_cols[0].metric("Unified NAV Rows", str(len(nav_raw)))
    ledger_cols[1].metric("Unified Position Rows", str(len(position_history)))
    ledger_cols[2].metric("Paper DB Positions", str(len(legacy_positions)))
    ledger_cols[3].metric("Review Rows", str(len(reviews)))
    ledger_cols[4].metric("Ticket Rows", str(len(orders)))

    st.subheader("Ledger Paths")
    render_dark_table(
        pd.DataFrame(
            [
                {"Ledger": "Unified account ledger", "Path": display_path(data["account_ledger"])},
                {"Ledger": "Paper trading ledger", "Path": display_path(data["paper_ledger"])},
            ]
        ),
    )

    st.subheader("QMT Connector Contracts")
    render_dark_table(qmt_connector_contract_frame(settings), max_height_px=260)

    st.subheader("QMT Audit Logs")
    render_qmt_audit_panel(settings, limit=15)

    st.subheader("Legacy Paper Positions")
    if legacy_positions.empty:
        st.info("No rows are available in the paper trading ledger position table.")
    else:
        render_dark_table(legacy_positions, max_height_px=420)

    st.subheader("Top Symbol History")
    if symbol_history.empty:
        st.info("No symbol-level paper position history is available yet.")
    else:
        latest_symbols = (
            symbol_history.sort_values("date")
            .groupby("symbol")
            .tail(1)
            .assign(abs_value=lambda frame: frame["market_value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(10)["symbol"]
            .tolist()
        )
        chart = symbol_history[symbol_history["symbol"].isin(latest_symbols)]
        if chart.empty:
            st.info("No top-symbol history is available yet.")
        else:
            render_dark_line_chart(
                chart.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum").fillna(0.0),
                yaxis_title="Market Value",
            )

with health_tab:
    st.subheader("Paper Runtime Health")
    if health.empty:
        st.info("No health checks are available.")
    else:
        render_dark_table(health, max_height_px=420)

    st.subheader("Runtime Gates")
    render_dark_table(
        pd.DataFrame(
            [
                {"Gate": "Trading Mode", "Value": settings.trading_mode},
                {"Gate": "Live Trading", "Value": str(settings.allow_live_trading).lower()},
                {"Gate": "Paper Trading", "Value": str(settings.allow_paper_trading).lower()},
                {"Gate": "Paper Submit", "Value": str(settings.allow_paper_order_submit).lower()},
                {"Gate": "Max Order Notional", "Value": money(settings.paper_max_order_notional)},
                {"Gate": "Max Daily Notional", "Value": money(settings.paper_max_daily_notional)},
                {"Gate": "Allowed Asset Classes", "Value": ", ".join(settings.paper_allowed_asset_classes)},
                {"Gate": "Options Enabled", "Value": str(settings.paper_options_enabled).lower()},
                {"Gate": "QMT Paper Submit", "Value": str(settings.allow_qmt_paper_order_submit).lower()},
                {"Gate": "QMT Submit URL", "Value": settings.qmt_submit_connector_url},
                {"Gate": "QMT HMAC Signing", "Value": "configured" if settings.qmt_request_signing_secret else "missing"},
            ]
        ),
    )

    st.subheader("QMT Runtime Gates")
    render_qmt_safety_panel(settings, max_height_px=360)
