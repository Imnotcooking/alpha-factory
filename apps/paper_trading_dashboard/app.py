"""Paper trading pre-flight dashboard."""

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
from oqp.brokers import (  # noqa: E402
    get_broker_adapter,
    get_broker_profile_config,
)
from oqp.config import load_settings  # noqa: E402
from oqp.contracts import (  # noqa: E402
    LoadedStrategyCandidate,
    load_strategy_candidate_artifacts,
    strategy_candidate_directory,
)
from oqp.data import (  # noqa: E402
    get_fundamentals_adapter,
    get_market_data_adapter,
    get_options_adapter,
)
from oqp.execution import (  # noqa: E402
    LoadedTradeProposal,
    TradeProposal,
    evaluate_trade_proposal,
    load_trade_proposal_artifacts,
    trade_proposal_directory,
)
from oqp.paper_trading import (  # noqa: E402
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_nav,
    load_latest_paper_orders,
    load_latest_paper_positions,
    paper_order_notional_today,
    review_paper_execution_proposal,
)
from apps.broker_monitor import (  # noqa: E402
    connect_readonly_snapshot,
    render_account_metrics,
    render_broker_health_json,
    render_cash_table,
    render_open_orders_table,
    render_positions_table,
    yes_no,
)


st.set_page_config(page_title="Paper Trading", layout="wide", page_icon="PT")


def present(value: str | None) -> str:
    return "present" if value else "missing"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def classify_data_health(ok: bool, implemented: bool) -> str:
    if ok and implemented:
        return "configured"
    if ok:
        return "stub"
    return "missing"


def metadata_flag(metadata: dict[str, Any], key: str) -> bool:
    return bool(metadata.get(key))


def data_health_row(role: str, adapter_name: str, health: Any) -> dict[str, Any]:
    metadata = dict(health.metadata or {})
    implemented = metadata_flag(metadata, "implemented")
    configured = bool(health.ok)

    return {
        "Layer": role,
        "Adapter": adapter_name,
        "Status": classify_data_health(configured, implemented),
        "Configured": yes_no(configured),
        "Implemented": yes_no(implemented),
        "Message": health.message or "",
    }


def safe_data_health(role: str, adapter_name: str, adapter: Any) -> dict[str, Any]:
    try:
        return data_health_row(role, adapter_name, adapter.healthcheck())
    except Exception as exc:
        return {
            "Layer": role,
            "Adapter": adapter_name,
            "Status": "error",
            "Configured": "no",
            "Implemented": "no",
            "Message": str(exc),
        }


def gate_row(check: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "Check": check,
        "Status": "pass" if passed else "blocked",
        "Detail": detail,
    }


def format_money(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:,.2f}"


def format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_signed_money(value: float | None) -> str:
    if value is None:
        return ""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:,.2f}"


def proposal_label(loaded: LoadedTradeProposal) -> str:
    proposal = loaded.proposal
    return f"{proposal.proposal_id} ({display_path(loaded.path)})"


def proposal_row(
    loaded: LoadedTradeProposal,
    *,
    reviewed_proposal_ids: set[str] | None = None,
) -> dict[str, Any]:
    proposal = loaded.proposal
    reviewed_ids = reviewed_proposal_ids or set()
    return {
        "Proposal ID": proposal.proposal_id,
        "Safety Review": (
            "reviewed" if proposal.proposal_id in reviewed_ids else "pending"
        ),
        "Source": proposal.source,
        "Status": proposal.status.value,
        "Intents": len(proposal.intents),
        "Estimated Notional": format_money(proposal.estimated_notional),
        "Paper Only": yes_no(proposal.paper_only),
        "Strategy": proposal.strategy_id or "",
        "Research Run": proposal.research_run_id or "",
        "Created": proposal.created_at.isoformat(timespec="seconds"),
        "Artifact": display_path(loaded.path),
    }


def parse_review_checks(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(decoded, list):
        return []
    return [dict(item) for item in decoded if isinstance(item, dict)]


def parse_metadata(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def review_blockers(checks: list[dict[str, Any]]) -> str:
    blockers = [
        str(check.get("name", ""))
        for check in checks
        if not bool(check.get("passed")) and check.get("severity") == "block"
    ]
    return ", ".join(item for item in blockers if item)


def review_history_row(row: pd.Series) -> dict[str, Any]:
    checks = parse_review_checks(row.get("checks_json"))
    blockers = review_blockers(checks)
    return {
        "Reviewed": str(row.get("reviewed_at", "")),
        "Proposal ID": str(row.get("proposal_id", "")),
        "Decision": str(row.get("decision", "")),
        "Estimated Notional": format_money(
            coerce_float(row.get("estimated_notional"))
        ),
        "Orders": int(coerce_float(row.get("order_count")) or 0),
        "Blocked Reasons": blockers,
        "Message": str(row.get("message", "")),
    }


def review_check_rows(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "Check": str(check.get("name", "")),
            "Status": "pass" if bool(check.get("passed")) else "blocked",
            "Severity": str(check.get("severity", "")),
            "Detail": str(check.get("detail", "")),
        }
        for check in checks
    ]


def paper_position_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Asset Type",
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
        "asset_type",
        "quantity",
        "market_price",
        "market_value",
        "unrealized_pnl",
        "currency",
        "as_of",
    ]
    display_df = df.reindex(columns=columns).copy()
    return display_df.rename(
        columns={
            "symbol": "Symbol",
            "asset_type": "Asset Type",
            "quantity": "Quantity",
            "market_price": "Market Price",
            "market_value": "Market Value",
            "unrealized_pnl": "Unrealized P&L",
            "currency": "Currency",
            "as_of": "As Of",
        }
    )


def paper_order_ticket_display(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Ticket ID",
        "Created",
        "Status",
        "Proposal",
        "Review",
        "Symbol",
        "Side",
        "Quantity",
        "Type",
        "Price",
        "Strategy",
        "Est. Notional",
        "Submit Enabled",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for row in df.to_dict("records"):
        metadata = parse_metadata(row.get("metadata_json"))
        rows.append(
            {
                "Ticket ID": row.get("order_id"),
                "Created": row.get("created_at"),
                "Status": row.get("status"),
                "Proposal": metadata.get("proposal_id", ""),
                "Review": metadata.get("review_id", ""),
                "Symbol": row.get("symbol"),
                "Side": row.get("side"),
                "Quantity": row.get("quantity"),
                "Type": row.get("order_type"),
                "Price": row.get("limit_price"),
                "Strategy": row.get("strategy_id") or "",
                "Est. Notional": metadata.get("estimated_notional"),
                "Submit Enabled": yes_no(bool(metadata.get("broker_submit_enabled"))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def intent_row(intent: Any) -> dict[str, Any]:
    return {
        "Symbol": intent.instrument.symbol,
        "Asset Class": intent.instrument.asset_class.value,
        "Side": intent.side.value,
        "Quantity": intent.quantity,
        "Type": intent.order_type.value,
        "Limit": format_money(intent.limit_price),
        "Stop": format_money(intent.stop_price),
        "Reference": format_money(intent.reference_price),
        "Est. Notional": format_money(intent.estimated_notional),
        "TIF": intent.time_in_force,
        "Confidence": "" if intent.confidence is None else f"{intent.confidence:.2f}",
        "Signal": intent.signal_id or "",
    }


def strategy_candidate_queue_detail(candidate: Any) -> str:
    if candidate.can_enter_paper_queue:
        return "ready for guarded paper intake"
    if candidate.promotion_status.value == "paper_running":
        return "already marked paper_running"
    if candidate.promotion_status.value != "paper_candidate":
        return f"status is {candidate.promotion_status.value}"

    blockers = []
    if not candidate.safety_limits.paper_only:
        blockers.append("paper_only is false")
    if candidate.safety_limits.allow_live_trading:
        blockers.append("live trading is allowed")
    if candidate.target_market_vertical != candidate.tested_market_vertical:
        blockers.append("target market differs from tested market")
    if candidate.instrument_mapping_required:
        blockers.append("instrument mapping required")
    return "; ".join(blockers) or "requires review"


def strategy_candidate_row(loaded: LoadedStrategyCandidate) -> dict[str, Any]:
    candidate = loaded.candidate
    return {
        "Candidate ID": candidate.candidate_id,
        "Strategy": candidate.strategy_id,
        "Intake State": candidate.intake_state_label,
        "Status": candidate.promotion_status.value,
        "Market Status": candidate.market_scoped_status,
        "Native Market": candidate.native_market_vertical,
        "Tested Market": candidate.tested_market_vertical,
        "Target Market": candidate.target_market_vertical,
        "Intended Markets": ", ".join(candidate.intended_market_verticals),
        "Research Run": candidate.research_run_id or "",
        "Geometry": candidate.evaluation_geometry or "",
        "Metric": candidate.ic_metric or "",
        "Holdout IC": format_metric(candidate.metrics.holdout_ic),
        "Sharpe": format_metric(candidate.metrics.sharpe_ratio, digits=2),
        "P-Value": format_metric(candidate.metrics.metric_p_value),
        "Mapping": yes_no(candidate.instrument_mapping_required),
        "Paper Queue Eligible": yes_no(candidate.can_enter_paper_queue),
        "Queue Detail": strategy_candidate_queue_detail(candidate),
        "Artifact": display_path(loaded.path),
    }


settings = load_settings()

market_adapter = get_market_data_adapter("yahoo", settings=settings)
fundamentals_adapter = get_fundamentals_adapter("fmp", settings=settings)
massive_options_adapter = get_options_adapter("massive", settings=settings)
polygon_options_adapter = get_options_adapter("polygon", settings=settings)

broker_config = get_broker_profile_config("ibkr_paper_readonly", settings=settings)
broker = get_broker_adapter("ibkr", settings=settings)
snapshot = connect_readonly_snapshot(broker, broker_config)
broker_health = snapshot["health"]
broker_connected = snapshot["connected"]
account_summary = snapshot["account_summary"]
cash_balances = snapshot["cash_balances"]
positions = snapshot["positions"]
open_orders = snapshot["open_orders"]
snapshot_error = snapshot["snapshot_error"]
proposal_result = load_trade_proposal_artifacts(trade_proposal_directory(settings))
candidate_result = load_strategy_candidate_artifacts(
    strategy_candidate_directory(settings),
    max_files=50,
)
paper_ledger_path = default_paper_trading_ledger_path()
account_ledger_path = default_account_ledger_path()
paper_account_nav_df = load_latest_account_nav(account_ledger_path, environment="paper")
paper_account_nav_history = account_nav_drawdowns(
    load_account_nav_history(account_ledger_path, environment="paper")
)
paper_account_positions_df = load_latest_account_positions(
    account_ledger_path,
    environment="paper",
)
paper_account_events_df = load_account_trade_events(
    account_ledger_path,
    environment="paper",
    limit=50,
)
paper_nav_df = load_latest_paper_nav(paper_ledger_path)
paper_positions_df = load_latest_paper_positions(paper_ledger_path)
paper_reviews_df = load_latest_paper_execution_reviews(paper_ledger_path, limit=25)
paper_orders_df = load_latest_paper_orders(paper_ledger_path, limit=50)
paper_daily_notional_used = paper_order_notional_today(paper_ledger_path)
reviewed_proposal_ids = (
    set(paper_reviews_df["proposal_id"].dropna().astype(str))
    if "proposal_id" in paper_reviews_df.columns
    else set()
)
unreviewed_proposal_count = sum(
    loaded.proposal.proposal_id not in reviewed_proposal_ids
    for loaded in proposal_result.loaded
)

preflight_proposal = TradeProposal(
    proposal_id="paper-preflight",
    source="paper_dashboard",
    intents=(),
    notes="System preflight proposal with no order intents.",
)

data_rows = [
    safe_data_health("Market data", "Yahoo", market_adapter),
    safe_data_health("Fundamentals", "FMP", fundamentals_adapter),
    safe_data_health("Options data", "Massive", massive_options_adapter),
    safe_data_health("Options greeks", "Massive snapshot", polygon_options_adapter),
]
data_health_df = pd.DataFrame(data_rows)

fmp_ready = data_health_df.loc[data_health_df["Adapter"] == "FMP", "Status"].isin(
    ["configured", "stub"]
).any()
options_ready = data_health_df.loc[
    data_health_df["Layer"].isin(["Options data", "Options greeks"]), "Status"
].isin(["configured", "stub"]).any()
paper_mode = settings.trading_mode.lower() == "paper"
live_disabled = not settings.allow_live_trading
broker_readonly = broker_config.readonly

gate_rows = [
    gate_row("Trading mode", paper_mode, settings.trading_mode.lower()),
    gate_row("Live trading disabled", live_disabled, yes_no(live_disabled)),
    gate_row("Broker read-only", broker_readonly, yes_no(broker_readonly)),
    gate_row("Fundamentals data", fmp_ready, "FMP key " + present(settings.fmp_api_key)),
    gate_row(
        "Options data",
        options_ready,
        "Massive key "
        + present(settings.massive_api_key or settings.options_api_key)
        + ", legacy Polygon alias "
        + present(settings.polygon_api_key),
    ),
    gate_row(
        "IBKR connection",
        broker_connected,
        broker_health.status.value,
    ),
    gate_row("Order placement", False, "disabled by read-only adapter"),
]
gate_df = pd.DataFrame(gate_rows)
execution_ready = bool(gate_df["Status"].eq("pass").all())
latest_paper_account_nav = (
    None if paper_account_nav_df.empty else paper_account_nav_df.iloc[0]
)
paper_account_nav_value = (
    None
    if latest_paper_account_nav is None
    else coerce_float(latest_paper_account_nav.get("net_liquidation"))
)
paper_account_daily_pnl = (
    None
    if latest_paper_account_nav is None
    else coerce_float(latest_paper_account_nav.get("daily_pnl"))
)
paper_account_cash_value = (
    None
    if latest_paper_account_nav is None
    else coerce_float(latest_paper_account_nav.get("cash"))
)
paper_account_position_count = (
    None
    if latest_paper_account_nav is None
    else int(coerce_float(latest_paper_account_nav.get("position_count")) or 0)
)
paper_account_as_of = (
    ""
    if latest_paper_account_nav is None
    else str(latest_paper_account_nav.get("as_of") or "")
)
latest_paper_nav = None if paper_nav_df.empty else paper_nav_df.iloc[0]
legacy_paper_nav_value = (
    None if latest_paper_nav is None else coerce_float(latest_paper_nav.get("net_liquidation"))
)
legacy_paper_daily_pnl = (
    None if latest_paper_nav is None else coerce_float(latest_paper_nav.get("daily_pnl"))
)
legacy_paper_cash_value = (
    None if latest_paper_nav is None else coerce_float(latest_paper_nav.get("cash"))
)
legacy_paper_position_count = (
    0 if latest_paper_nav is None else int(coerce_float(latest_paper_nav.get("position_count")) or 0)
)
paper_nav_value = (
    paper_account_nav_value
    if paper_account_nav_value is not None
    else legacy_paper_nav_value
)
paper_daily_pnl = (
    paper_account_daily_pnl
    if paper_account_daily_pnl is not None
    else legacy_paper_daily_pnl
)
paper_cash_value = (
    paper_account_cash_value
    if paper_account_cash_value is not None
    else legacy_paper_cash_value
)
paper_position_count = (
    paper_account_position_count
    if paper_account_position_count is not None
    else legacy_paper_position_count
)
paper_nav_source = (
    "unified account ledger"
    if latest_paper_account_nav is not None
    else "paper trading ledger"
)


st.title("Paper Trading")
st.caption("IBKR paper execution cockpit")

summary_cols = st.columns(6)
summary_cols[0].metric("Execution Gate", "unlocked" if execution_ready else "locked")
summary_cols[1].metric("Snapshot", "ready" if account_summary else "offline")
summary_cols[2].metric("IBKR", broker_health.status.value)
summary_cols[3].metric("Paper NAV", format_money(paper_nav_value) or "missing")
summary_cols[4].metric("Review Queue", str(unreviewed_proposal_count))
summary_cols[5].metric("Account Events", str(len(paper_account_events_df)))

st.error("Execution locked. Order placement remains unavailable.")

if account_summary:
    render_account_metrics(account_summary)
elif snapshot_error:
    st.warning(f"Connected to IBKR, but snapshot fetch failed: {snapshot_error}")
else:
    st.info(
        "Log in to TWS or IB Gateway locally, enable API socket access, and keep "
        "this app pointed at the paper port. Do not store your IBKR username or "
        "password in this repository."
    )

st.subheader("Unified Paper Account")
ledger_metric_cols = st.columns(5)
ledger_metric_cols[0].metric("Recorded NAV", format_money(paper_nav_value) or "missing")
ledger_metric_cols[1].metric("Daily P&L", format_signed_money(paper_daily_pnl) or "missing")
ledger_metric_cols[2].metric("Cash", format_money(paper_cash_value) or "missing")
ledger_metric_cols[3].metric("Ledger Positions", str(paper_position_count))
ledger_metric_cols[4].metric(
    "Snapshot As Of",
    paper_account_as_of or "missing",
)
st.caption(f"Account ledger path: {display_path(account_ledger_path)}")

if paper_account_nav_history.empty:
    st.info("No unified paper account NAV history has been recorded yet.")
else:
    account_chart_left, account_chart_right = st.columns([1.2, 1])
    with account_chart_left:
        st.line_chart(
            paper_account_nav_history.set_index("date")[
                ["net_liquidation", "equity_peak"]
            ]
        )
    with account_chart_right:
        st.line_chart(paper_account_nav_history.set_index("date")[["drawdown"]])

account_left, account_right = st.columns([1.2, 1])
with account_left:
    st.markdown("#### Unified Account Positions")
    account_positions_table = account_positions_display(paper_account_positions_df)
    if account_positions_table.empty:
        st.info("No unified paper account positions have been recorded yet.")
    else:
        st.dataframe(
            account_positions_table,
            use_container_width=True,
            hide_index=True,
        )

with account_right:
    st.markdown("#### Unified Asset Mix")
    st.dataframe(
        account_asset_summary(paper_account_positions_df),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Unified Paper Account Events")
if paper_account_events_df.empty:
    st.info("No paper account events have been recorded yet.")
else:
    event_left, event_right = st.columns([1.4, 1])
    with event_left:
        st.dataframe(
            account_trade_events_display(paper_account_events_df),
            use_container_width=True,
            hide_index=True,
        )
    with event_right:
        st.dataframe(
            account_trade_event_summary(paper_account_events_df),
            use_container_width=True,
            hide_index=True,
        )
st.caption(f"NAV source: {paper_nav_source}")

st.subheader("Paper Trading Ledger")
paper_ledger_metric_cols = st.columns(6)
paper_ledger_metric_cols[0].metric(
    "Legacy NAV",
    format_money(legacy_paper_nav_value) or "missing",
)
paper_ledger_metric_cols[1].metric(
    "Legacy Daily P&L",
    format_signed_money(legacy_paper_daily_pnl) or "missing",
)
paper_ledger_metric_cols[2].metric(
    "Legacy Cash",
    format_money(legacy_paper_cash_value) or "missing",
)
paper_ledger_metric_cols[3].metric(
    "Legacy Positions",
    str(legacy_paper_position_count),
)
paper_ledger_metric_cols[4].metric(
    "Daily Notional Used",
    format_money(paper_daily_notional_used) or "0.00",
)
paper_ledger_metric_cols[5].metric(
    "Dry-Run Tickets",
    str(len(paper_orders_df[paper_orders_df["status"].eq("dry_run")]))
    if not paper_orders_df.empty and "status" in paper_orders_df.columns
    else "0",
)
st.caption(f"Paper trading ledger path: {display_path(paper_ledger_path)}")

st.markdown("#### Dry-Run Order Tickets")
order_ticket_display = paper_order_ticket_display(paper_orders_df)
if order_ticket_display.empty:
    st.info("No dry-run paper order tickets have been created yet.")
else:
    st.dataframe(order_ticket_display, use_container_width=True, hide_index=True)

ledger_left, ledger_right = st.columns([1.2, 1])
with ledger_left:
    st.markdown("#### Latest Ledger Positions")
    ledger_positions_display = paper_position_display(paper_positions_df)
    if ledger_positions_display.empty:
        st.info("No paper positions have been recorded yet.")
    else:
        st.dataframe(
            ledger_positions_display,
            use_container_width=True,
            hide_index=True,
        )

with ledger_right:
    st.markdown("#### Safety Review History")
    review_history_df = pd.DataFrame(
        [review_history_row(row) for _, row in paper_reviews_df.iterrows()]
    )
    if review_history_df.empty:
        review_history_df = pd.DataFrame(
            columns=[
                "Reviewed",
                "Proposal ID",
                "Decision",
                "Estimated Notional",
                "Orders",
                "Blocked Reasons",
                "Message",
            ]
        )
        st.info("No paper execution safety reviews have been recorded yet.")
    else:
        st.dataframe(review_history_df, use_container_width=True, hide_index=True)

if not paper_reviews_df.empty:
    review_labels = [
        f"{row.proposal_id} ({row.reviewed_at})"
        for row in paper_reviews_df.itertuples(index=False)
    ]
    selected_review_label = st.selectbox(
        "Safety review detail",
        options=review_labels,
        index=0,
    )
    selected_review_index = review_labels.index(selected_review_label)
    selected_review = paper_reviews_df.iloc[selected_review_index]
    selected_checks = parse_review_checks(selected_review.get("checks_json"))
    checks_df = pd.DataFrame(review_check_rows(selected_checks))
    st.dataframe(checks_df, use_container_width=True, hide_index=True)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Adapter Health")
    st.dataframe(
        data_health_df,
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Runtime Config")
    config_df = pd.DataFrame(
        [
            {"Setting": "IBKR host", "Value": settings.ibkr_host},
            {"Setting": "IBKR profile", "Value": broker_config.metadata.get("profile", "")},
            {"Setting": "IBKR paper port", "Value": str(broker_config.port)},
            {"Setting": "IBKR client id", "Value": str(broker_config.client_id)},
            {"Setting": "Live trading allowed", "Value": yes_no(settings.allow_live_trading)},
            {
                "Setting": "Paper trading allowed",
                "Value": yes_no(settings.allow_paper_trading),
            },
            {
                "Setting": "Paper max order notional",
                "Value": format_money(settings.paper_max_order_notional),
            },
            {
                "Setting": "Paper daily notional cap",
                "Value": format_money(settings.paper_max_daily_notional),
            },
            {
                "Setting": "Paper asset classes",
                "Value": ", ".join(settings.paper_allowed_asset_classes),
            },
            {
                "Setting": "Paper options enabled",
                "Value": yes_no(settings.paper_options_enabled),
            },
            {
                "Setting": "Option underlyings",
                "Value": ", ".join(settings.paper_option_allowed_underlyings) or "none",
            },
            {
                "Setting": "Option strategies",
                "Value": ", ".join(settings.paper_option_allowed_strategies) or "none",
            },
            {
                "Setting": "Option max contracts",
                "Value": (
                    "none"
                    if settings.paper_option_max_contracts is None
                    else f"{settings.paper_option_max_contracts:g}"
                ),
            },
            {
                "Setting": "Option max premium",
                "Value": format_money(settings.paper_option_max_premium),
            },
            {
                "Setting": "Option max defined risk",
                "Value": format_money(settings.paper_option_max_defined_risk),
            },
            {
                "Setting": "Option max spread width",
                "Value": (
                    "none"
                    if settings.paper_option_max_spread_width is None
                    else f"{settings.paper_option_max_spread_width:g}"
                ),
            },
            {"Setting": "FMP key", "Value": present(settings.fmp_api_key)},
            {
                "Setting": "Massive or options key",
                "Value": present(settings.massive_api_key or settings.options_api_key),
            },
            {
                "Setting": "Massive / legacy Polygon key",
                "Value": present(settings.polygon_api_key),
            },
            {
                "Setting": "Massive flat files",
                "Value": yes_no(
                    bool(
                        settings.massive_flat_files_access_key_id
                        and settings.massive_flat_files_secret_access_key
                    )
                ),
            },
            {"Setting": "Data root", "Value": display_path(settings.data_root)},
            {
                "Setting": "Artifact root",
                "Value": display_path(settings.artifact_root),
            },
        ]
    )
    st.dataframe(config_df, use_container_width=True, hide_index=True)

st.subheader("Execution Gate")
st.dataframe(gate_df, use_container_width=True, hide_index=True)

st.subheader("Strategy Candidate Intake")
CANDIDATE_COLUMNS = [
    "Candidate ID",
    "Strategy",
    "Intake State",
    "Status",
    "Market Status",
    "Native Market",
    "Tested Market",
    "Target Market",
    "Intended Markets",
    "Research Run",
    "Geometry",
    "Metric",
    "Holdout IC",
    "Sharpe",
    "P-Value",
    "Mapping",
    "Paper Queue Eligible",
    "Queue Detail",
    "Artifact",
]
candidate_df = pd.DataFrame(
    [strategy_candidate_row(loaded) for loaded in candidate_result.loaded]
)
if candidate_df.empty:
    candidate_df = pd.DataFrame(columns=CANDIDATE_COLUMNS)
else:
    candidate_df = candidate_df.reindex(columns=CANDIDATE_COLUMNS)

queue_candidate_df = candidate_df[
    candidate_df["Paper Queue Eligible"].eq("yes")
].copy()
snapshot_candidate_df = candidate_df[
    ~candidate_df["Paper Queue Eligible"].eq("yes")
].copy()

intake_cols = st.columns(3)
intake_cols[0].metric("Candidate Snapshots", str(len(candidate_df)))
intake_cols[1].metric("Paper Queue Eligible", str(len(queue_candidate_df)))
intake_cols[2].metric("Review / History", str(len(snapshot_candidate_df)))
st.caption(
    "Exported research snapshots are visible here for review. Only rows marked "
    "Paper Queue Eligible may feed paper-trading proposals."
)

st.markdown("#### Paper Queue Eligible")
if queue_candidate_df.empty:
    st.info("No exported candidate is currently eligible for the paper queue.")
else:
    st.dataframe(queue_candidate_df, use_container_width=True, hide_index=True)

st.markdown("#### Research Candidate Snapshots")
st.dataframe(snapshot_candidate_df, use_container_width=True, hide_index=True)
st.caption(f"Artifact directory: {display_path(candidate_result.directory)}")

if candidate_result.issues:
    with st.expander("Strategy Candidate Artifact Issues", expanded=False):
        candidate_issue_df = pd.DataFrame(
            [
                {
                    "Artifact": display_path(issue.path),
                    "Issue": issue.message,
                }
                for issue in candidate_result.issues
            ]
        )
        st.dataframe(candidate_issue_df, use_container_width=True, hide_index=True)

proposal_left, proposal_right = st.columns([1.2, 1])

selected_proposal = preflight_proposal
selected_loaded: LoadedTradeProposal | None = None

with proposal_left:
    st.subheader("Draft Trade Proposals")
    proposal_df = pd.DataFrame(
        [
            proposal_row(loaded, reviewed_proposal_ids=reviewed_proposal_ids)
            for loaded in proposal_result.loaded
        ]
    )
    if proposal_df.empty:
        proposal_df = pd.DataFrame(
            columns=[
                "Proposal ID",
                "Safety Review",
                "Source",
                "Status",
                "Intents",
                "Estimated Notional",
                "Paper Only",
                "Strategy",
                "Research Run",
                "Created",
                "Artifact",
            ]
        )
    st.dataframe(proposal_df, use_container_width=True, hide_index=True)
    st.caption(f"Artifact directory: {display_path(proposal_result.directory)}")

    if proposal_result.loaded:
        proposal_lookup = {
            proposal_label(loaded): loaded for loaded in proposal_result.loaded
        }
        selected_label = st.selectbox(
            "Guardrail target",
            options=list(proposal_lookup),
            index=0,
        )
        selected_loaded = proposal_lookup[selected_label]
        selected_proposal = selected_loaded.proposal
        intent_df = pd.DataFrame(
            [intent_row(intent) for intent in selected_proposal.intents]
        )
        st.dataframe(intent_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No draft proposals pending.")

    if proposal_result.issues:
        with st.expander("Artifact Issues", expanded=False):
            issue_df = pd.DataFrame(
                [
                    {
                        "Artifact": display_path(issue.path),
                        "Issue": issue.message,
                    }
                    for issue in proposal_result.issues
                ]
            )
            st.dataframe(issue_df, use_container_width=True, hide_index=True)

with proposal_right:
    st.subheader("Proposal Guardrails")
    if selected_loaded:
        st.caption(proposal_label(selected_loaded))
    else:
        st.caption("paper-preflight")
    ui_gate_tab, paper_safety_tab = st.tabs(["UI Gate", "Paper Safety"])

    with ui_gate_tab:
        guardrail_report = evaluate_trade_proposal(
            selected_proposal,
            settings=settings,
            broker_config=broker_config,
            account_summary=account_summary,
            broker_connected=broker_connected,
            order_placement_enabled=False,
        )
        guardrail_df = pd.DataFrame(
            [
                {
                    "Check": check.name,
                    "Status": "pass" if check.passed else "blocked",
                    "Severity": check.severity.value,
                    "Detail": check.detail,
                }
                for check in guardrail_report.checks
            ]
        )
        st.dataframe(guardrail_df, use_container_width=True, hide_index=True)

    with paper_safety_tab:
        safety_review = review_paper_execution_proposal(
            selected_proposal,
            settings=settings,
            broker_config=broker_config,
            daily_notional_used=paper_daily_notional_used,
        )
        safety_cols = st.columns(3)
        safety_cols[0].metric("Decision", safety_review.decision.value)
        safety_cols[1].metric(
            "Estimated Notional",
            format_money(safety_review.estimated_notional) or "unknown",
        )
        safety_cols[2].metric("Orders", str(safety_review.order_count))
        safety_df = pd.DataFrame(
            [
                {
                    "Check": check.name,
                    "Status": "pass" if check.passed else "blocked",
                    "Severity": check.severity.value,
                    "Detail": check.detail,
                }
                for check in safety_review.checks
            ]
        )
        st.dataframe(safety_df, use_container_width=True, hide_index=True)

snapshot_left, snapshot_right = st.columns([1.4, 1])

with snapshot_left:
    st.subheader("Read-Only Positions")
    render_positions_table(positions)

with snapshot_right:
    st.subheader("Cash Balances")
    render_cash_table(cash_balances)

st.subheader("Read-Only Open Orders")
render_open_orders_table(open_orders)

with st.expander("Broker Health Payload", expanded=False):
    render_broker_health_json(broker_health)
