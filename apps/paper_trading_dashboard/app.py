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
    PaperOrderTicketStatus,
    PaperStrategyStatus,
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_nav,
    load_latest_paper_orders,
    load_latest_paper_positions,
    load_paper_strategy_record,
    load_paper_strategy_registry,
    paper_order_notional_today,
    review_paper_execution_proposal,
    review_paper_order_submission,
    review_paper_strategy_gate,
    set_paper_order_ticket_approval,
    upsert_paper_strategy_from_candidate,
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


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def paper_strategy_registry_display(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Status",
        "Market",
        "Candidate",
        "Research Run",
        "Max Order",
        "Max Daily",
        "Allowed Symbols",
        "Rebalance",
        "Kill Switch",
        "Approved By",
        "Approved At",
        "Notes",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                "Strategy": row.get("strategy_id", ""),
                "Status": row.get("status", ""),
                "Market": row.get("market_vertical", ""),
                "Candidate": row.get("candidate_id", ""),
                "Research Run": row.get("research_run_id", ""),
                "Max Order": row.get("max_order_notional"),
                "Max Daily": row.get("max_daily_notional"),
                "Allowed Symbols": ", ".join(
                    parse_json_list(row.get("allowed_symbols_json"))
                ),
                "Rebalance": row.get("rebalance_frequency", ""),
                "Kill Switch": yes_no(bool(row.get("kill_switch"))),
                "Approved By": row.get("approved_by", ""),
                "Approved At": row.get("approved_at", ""),
                "Notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def count_status(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].astype(str).eq(value).sum())


def metadata_json_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        return parse_metadata(row.get("metadata_json"))
    return parse_metadata(row.get("metadata_json"))


def latest_review_by_proposal(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "proposal_id" not in df.columns:
        return {}
    lookup: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        proposal_id = str(row.get("proposal_id") or "").strip()
        if proposal_id and proposal_id not in lookup:
            lookup[proposal_id] = row
    return lookup


def ticket_summary_by_proposal(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    if df.empty:
        return summary
    for row in df.to_dict("records"):
        metadata = metadata_json_from_row(row)
        proposal_id = str(metadata.get("proposal_id") or "").strip()
        if not proposal_id:
            continue
        status = str(row.get("status") or "unknown")
        entry = summary.setdefault(
            proposal_id,
            {
                "count": 0,
                "statuses": {},
                "order_ids": [],
            },
        )
        entry["count"] += 1
        entry["statuses"][status] = int(entry["statuses"].get(status, 0)) + 1
        entry["order_ids"].append(str(row.get("order_id") or ""))
    return summary


def summarize_ticket_statuses(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "0"
    statuses = summary.get("statuses", {})
    return ", ".join(f"{status}={count}" for status, count in statuses.items()) or "0"


def latest_log_line(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as exc:
        return f"unreadable: {exc}"
    return next((line for line in reversed(lines) if line), "empty")


def paper_pipeline_rows(
    *,
    candidate_count: int,
    running_strategy_count: int,
    proposal_count: int,
    unreviewed_count: int,
    review_count: int,
    ready_review_count: int,
    blocked_review_count: int,
    dry_run_ticket_count: int,
    approved_ticket_count: int,
    rejected_ticket_count: int,
    runner_log_line: str,
    paper_trading_allowed: bool,
    order_submit_allowed: bool,
    broker_readonly: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "Layer": "Research exports",
            "State": "present" if candidate_count else "waiting",
            "Count": candidate_count,
            "Current Detail": "strategy candidate artifacts",
            "Next Gate": "paper queue eligibility",
        },
        {
            "Layer": "Paper strategy registry",
            "State": "armed" if running_strategy_count else "waiting",
            "Count": running_strategy_count,
            "Current Detail": "paper_running strategies",
            "Next Gate": "proposal strategy gate",
        },
        {
            "Layer": "Trade proposals",
            "State": "pending" if unreviewed_count else ("present" if proposal_count else "waiting"),
            "Count": proposal_count,
            "Current Detail": f"{unreviewed_count} not reviewed",
            "Next Gate": "runner scan",
        },
        {
            "Layer": "Paper strategy runner",
            "State": "scheduled",
            "Count": "",
            "Current Detail": runner_log_line,
            "Next Gate": "safety review",
        },
        {
            "Layer": "Safety reviews",
            "State": "ready" if ready_review_count else ("blocked" if blocked_review_count else "waiting"),
            "Count": review_count,
            "Current Detail": f"ready={ready_review_count}, blocked={blocked_review_count}",
            "Next Gate": "dry-run ticket creation",
        },
        {
            "Layer": "Dry-run tickets",
            "State": "waiting approval" if dry_run_ticket_count else ("approved" if approved_ticket_count else "waiting"),
            "Count": dry_run_ticket_count + approved_ticket_count + rejected_ticket_count,
            "Current Detail": (
                f"dry_run={dry_run_ticket_count}, "
                f"approved={approved_ticket_count}, rejected={rejected_ticket_count}"
            ),
            "Next Gate": "submission preflight",
        },
        {
            "Layer": "Broker submission",
            "State": "locked",
            "Count": "",
            "Current Detail": (
                f"paper_trading={yes_no(paper_trading_allowed)}, "
                f"paper_submit={yes_no(order_submit_allowed)}, "
                f"broker_readonly={yes_no(broker_readonly)}"
            ),
            "Next Gate": "future IBKR paper order sender",
        },
    ]


def proposal_automation_row(
    loaded: LoadedTradeProposal,
    *,
    reviews_by_proposal: dict[str, pd.Series],
    tickets_by_proposal: dict[str, dict[str, Any]],
    paper_ledger_path: Path,
) -> dict[str, Any]:
    proposal = loaded.proposal
    review = reviews_by_proposal.get(proposal.proposal_id)
    ticket_summary = tickets_by_proposal.get(proposal.proposal_id)
    strategy_gate = review_paper_strategy_gate(paper_ledger_path, proposal)
    review_decision = "" if review is None else str(review.get("decision", ""))
    ticket_count = int((ticket_summary or {}).get("count", 0))

    if ticket_count:
        current_stage = "ticketed"
        next_action = "human approval or rejection"
    elif review is not None:
        current_stage = f"reviewed: {review_decision}"
        next_action = (
            "inspect blocked reasons"
            if review_decision == "blocked"
            else "waiting for ticket creation"
        )
    elif strategy_gate.passed:
        current_stage = "waiting for runner"
        next_action = "runner will safety-review"
    else:
        current_stage = "skipped by strategy gate"
        next_action = "register, unpause, or change strategy"

    return {
        "Proposal": proposal.proposal_id,
        "Strategy": proposal.strategy_id or "",
        "Current Stage": current_stage,
        "Strategy Gate": "pass" if strategy_gate.passed else "blocked",
        "Safety Review": review_decision or "pending",
        "Tickets": summarize_ticket_statuses(ticket_summary),
        "Next Action": next_action,
        "Reason": (
            str(review.get("message", ""))
            if review is not None
            else strategy_gate.message
        ),
    }


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
paper_strategy_registry_df = load_paper_strategy_registry(paper_ledger_path, limit=100)
paper_daily_notional_used = paper_order_notional_today(paper_ledger_path)
paper_strategy_runner_log = REPO_ROOT / "logs" / "paper_strategy_runner.log"
reviewed_proposal_ids = (
    set(paper_reviews_df["proposal_id"].dropna().astype(str))
    if "proposal_id" in paper_reviews_df.columns
    else set()
)
unreviewed_proposal_count = sum(
    loaded.proposal.proposal_id not in reviewed_proposal_ids
    for loaded in proposal_result.loaded
)
reviews_by_proposal = latest_review_by_proposal(paper_reviews_df)
tickets_by_proposal = ticket_summary_by_proposal(paper_orders_df)
paper_running_strategy_count = count_status(
    paper_strategy_registry_df,
    "status",
    PaperStrategyStatus.RUNNING.value,
)
paper_paused_strategy_count = count_status(
    paper_strategy_registry_df,
    "status",
    PaperStrategyStatus.PAUSED.value,
)
paper_retired_strategy_count = count_status(
    paper_strategy_registry_df,
    "status",
    PaperStrategyStatus.RETIRED.value,
)
ready_review_count = count_status(paper_reviews_df, "decision", "ready")
blocked_review_count = count_status(paper_reviews_df, "decision", "blocked")
dry_run_ticket_count = count_status(
    paper_orders_df,
    "status",
    PaperOrderTicketStatus.DRY_RUN.value,
)
approved_ticket_count = count_status(
    paper_orders_df,
    "status",
    PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
)
rejected_ticket_count = count_status(
    paper_orders_df,
    "status",
    PaperOrderTicketStatus.REJECTED.value,
)
runner_log_line = latest_log_line(paper_strategy_runner_log)

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

st.subheader("Paper Automation Control Center")
automation_metric_cols = st.columns(6)
automation_metric_cols[0].metric("Paper Running", str(paper_running_strategy_count))
automation_metric_cols[1].metric("Paused", str(paper_paused_strategy_count))
automation_metric_cols[2].metric("Retired", str(paper_retired_strategy_count))
automation_metric_cols[3].metric("Pending Proposals", str(unreviewed_proposal_count))
automation_metric_cols[4].metric("Dry-Run Tickets", str(dry_run_ticket_count))
automation_metric_cols[5].metric(
    "Broker Submit",
    "enabled" if settings.allow_paper_order_submit and not broker_config.readonly else "locked",
)

pipeline_tab, proposal_board_tab, locks_tab = st.tabs(
    ["Pipeline", "Proposal Board", "Safety Locks"]
)

with pipeline_tab:
    pipeline_df = pd.DataFrame(
        paper_pipeline_rows(
            candidate_count=len(candidate_result.loaded),
            running_strategy_count=paper_running_strategy_count,
            proposal_count=len(proposal_result.loaded),
            unreviewed_count=unreviewed_proposal_count,
            review_count=len(paper_reviews_df),
            ready_review_count=ready_review_count,
            blocked_review_count=blocked_review_count,
            dry_run_ticket_count=dry_run_ticket_count,
            approved_ticket_count=approved_ticket_count,
            rejected_ticket_count=rejected_ticket_count,
            runner_log_line=runner_log_line,
            paper_trading_allowed=settings.allow_paper_trading,
            order_submit_allowed=settings.allow_paper_order_submit,
            broker_readonly=broker_config.readonly,
        )
    )
    st.dataframe(pipeline_df, use_container_width=True, hide_index=True)

with proposal_board_tab:
    proposal_automation_df = pd.DataFrame(
        [
            proposal_automation_row(
                loaded,
                reviews_by_proposal=reviews_by_proposal,
                tickets_by_proposal=tickets_by_proposal,
                paper_ledger_path=paper_ledger_path,
            )
            for loaded in proposal_result.loaded
        ]
    )
    if proposal_automation_df.empty:
        proposal_automation_df = pd.DataFrame(
            columns=[
                "Proposal",
                "Strategy",
                "Current Stage",
                "Strategy Gate",
                "Safety Review",
                "Tickets",
                "Next Action",
                "Reason",
            ]
        )
        st.info("No trade proposal artifacts are currently loaded.")
    else:
        st.dataframe(
            proposal_automation_df,
            use_container_width=True,
            hide_index=True,
        )

with locks_tab:
    locks_df = pd.DataFrame(
        [
            {
                "Gate": "Live trading",
                "State": "locked" if not settings.allow_live_trading else "open",
                "Detail": f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
            },
            {
                "Gate": "Paper proposal review",
                "State": "open" if settings.allow_paper_trading else "locked",
                "Detail": f"ALLOW_PAPER_TRADING={str(settings.allow_paper_trading).lower()}",
            },
            {
                "Gate": "Paper broker profile",
                "State": "read-only" if broker_config.readonly else "write-enabled",
                "Detail": str(broker_config.metadata.get("profile") or "ibkr_paper_readonly"),
            },
            {
                "Gate": "Paper order submit",
                "State": "locked" if not settings.allow_paper_order_submit else "armed",
                "Detail": (
                    f"ALLOW_PAPER_ORDER_SUBMIT={str(settings.allow_paper_order_submit).lower()}"
                ),
            },
            {
                "Gate": "Runner log",
                "State": "present" if paper_strategy_runner_log.exists() else "missing",
                "Detail": runner_log_line,
            },
        ]
    )
    st.dataframe(locks_df, use_container_width=True, hide_index=True)

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

pending_ticket_df = (
    paper_orders_df[paper_orders_df["status"].eq(PaperOrderTicketStatus.DRY_RUN.value)]
    if not paper_orders_df.empty and "status" in paper_orders_df.columns
    else pd.DataFrame()
)
if pending_ticket_df.empty:
    st.info("No dry-run ticket is waiting for human approval.")
else:
    st.markdown("#### Human Ticket Approval")
    ticket_lookup = {
        (
            f"{row.order_id} | {row.symbol} {row.side} "
            f"{format_metric(coerce_float(row.quantity), digits=4)} {row.order_type}"
        ): row
        for row in pending_ticket_df.itertuples(index=False)
    }
    with st.form("paper_ticket_approval_form"):
        selected_ticket_label = st.selectbox(
            "Ticket",
            options=list(ticket_lookup),
            index=0,
        )
        decision = st.selectbox(
            "Decision",
            options=[
                PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
                PaperOrderTicketStatus.REJECTED.value,
            ],
            index=0,
        )
        decision_by = st.text_input("Decision by", value="dashboard")
        reason = st.text_input("Reason", value="manual paper ticket review")
        confirmation = st.text_input(
            "Confirmation",
            value="",
            placeholder="APPROVE or REJECT",
        )
        submitted = st.form_submit_button("Record Decision")

    if submitted:
        expected = (
            "APPROVE"
            if decision == PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value
            else "REJECT"
        )
        if confirmation.strip().upper() != expected:
            st.error(f"Type {expected} to record this decision.")
        else:
            selected_ticket = ticket_lookup[selected_ticket_label]
            paper_account_id = (
                None
                if latest_paper_account_nav is None
                else str(latest_paper_account_nav.get("account_id") or "").strip()
            )
            try:
                result = set_paper_order_ticket_approval(
                    order_id=str(selected_ticket.order_id),
                    status=decision,
                    paper_ledger_path=paper_ledger_path,
                    account_ledger_path=account_ledger_path,
                    broker_config=broker_config,
                    account_id=paper_account_id or None,
                    decided_by=decision_by.strip() or "dashboard",
                    reason=reason.strip() or None,
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(result.message)
                st.caption("Broker submit enabled: false")
                st.rerun()

approved_ticket_df = (
    paper_orders_df[
        paper_orders_df["status"].eq(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value)
    ]
    if not paper_orders_df.empty and "status" in paper_orders_df.columns
    else pd.DataFrame()
)
st.markdown("#### Submission Preflight")
if approved_ticket_df.empty:
    st.info("No approved paper ticket is waiting for submission preflight.")
else:
    approved_ticket_lookup = {
        (
            f"{row['order_id']} | {row['symbol']} {row['side']} "
            f"{format_metric(coerce_float(row['quantity']), digits=4)} "
            f"{row['order_type']}"
        ): row
        for row in approved_ticket_df.to_dict("records")
    }
    selected_submit_label = st.selectbox(
        "Submission preflight ticket",
        options=list(approved_ticket_lookup),
        index=0,
    )
    submission_preflight = review_paper_order_submission(
        approved_ticket_lookup[selected_submit_label],
        settings=settings,
        broker_config=broker_config,
        strategy_record=load_paper_strategy_record(
            paper_ledger_path,
            str(approved_ticket_lookup[selected_submit_label].get("strategy_id") or ""),
        ),
    )
    submit_cols = st.columns(3)
    submit_cols[0].metric("Decision", submission_preflight.decision.value)
    submit_cols[1].metric("Ticket", submission_preflight.order_id)
    submit_cols[2].metric("Broker Submit", "disabled")
    st.caption(submission_preflight.message)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Check": check.name,
                    "Status": "pass" if check.passed else "blocked",
                    "Severity": check.severity,
                    "Detail": check.detail,
                }
                for check in submission_preflight.checks
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("This preflight does not submit IBKR orders.")

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
                "Setting": "Paper order submit allowed",
                "Value": yes_no(settings.allow_paper_order_submit),
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

st.markdown("#### Paper Strategy Registry")
registry_display = paper_strategy_registry_display(paper_strategy_registry_df)
if registry_display.empty:
    st.info("No strategy has been approved for paper running yet.")
else:
    st.dataframe(registry_display, use_container_width=True, hide_index=True)

eligible_candidate_lookup = {
    f"{loaded.candidate.strategy_id} | {loaded.candidate.candidate_id} | "
    f"{loaded.candidate.target_market_vertical}": loaded
    for loaded in candidate_result.loaded
    if loaded.candidate.can_enter_paper_queue
}
if eligible_candidate_lookup:
    st.markdown("#### Approve Strategy For Paper Running")
    with st.form("approve_strategy_for_paper_running_form"):
        selected_candidate_label = st.selectbox(
            "Candidate",
            options=list(eligible_candidate_lookup),
            index=0,
        )
        selected_candidate = eligible_candidate_lookup[selected_candidate_label].candidate
        default_max_order = (
            selected_candidate.safety_limits.max_order_notional
            or settings.paper_max_order_notional
            or 0.0
        )
        default_max_daily = settings.paper_max_daily_notional or 0.0
        max_order_notional = st.number_input(
            "Max order notional",
            min_value=0.0,
            value=float(default_max_order),
            step=500.0,
        )
        max_daily_notional = st.number_input(
            "Max daily notional",
            min_value=0.0,
            value=float(default_max_daily),
            step=1000.0,
        )
        allowed_symbols_text = st.text_input(
            "Allowed symbols",
            value=", ".join(settings.paper_allowed_symbols),
        )
        rebalance_frequency = st.text_input(
            "Rebalance frequency",
            value="manual",
        )
        approved_by = st.text_input("Approved by", value="dashboard")
        notes = st.text_input("Notes", value="paper strategy approval")
        confirmation = st.text_input(
            "Confirmation",
            value="",
            placeholder="PAPER",
        )
        approve_strategy = st.form_submit_button("Approve Strategy")

    if approve_strategy:
        if confirmation.strip().upper() != "PAPER":
            st.error("Type PAPER to approve this strategy for paper running.")
        else:
            loaded_candidate = eligible_candidate_lookup[selected_candidate_label]
            try:
                result = upsert_paper_strategy_from_candidate(
                    paper_ledger_path,
                    loaded_candidate.candidate,
                    status=PaperStrategyStatus.RUNNING,
                    max_order_notional=max_order_notional or None,
                    max_daily_notional=max_daily_notional or None,
                    allowed_symbols=tuple(
                        symbol.strip()
                        for symbol in allowed_symbols_text.split(",")
                        if symbol.strip()
                    ),
                    rebalance_frequency=rebalance_frequency.strip() or "manual",
                    approved_by=approved_by.strip() or "dashboard",
                    notes=notes.strip() or None,
                    source_artifact=display_path(loaded_candidate.path),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"{result.strategy_id} approved as {result.status.value} "
                    f"for {result.market_vertical}."
                )
                st.rerun()

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
        strategy_gate = review_paper_strategy_gate(
            paper_ledger_path,
            selected_proposal,
        )
        st.markdown("##### Paper Strategy Gate")
        gate_cols = st.columns(3)
        gate_cols[0].metric(
            "Strategy Gate",
            "pass" if strategy_gate.passed else "blocked",
        )
        gate_cols[1].metric("Strategy", strategy_gate.strategy_id or "missing")
        gate_cols[2].metric(
            "Registry",
            (
                str(strategy_gate.record.get("status"))
                if strategy_gate.record
                else "missing"
            ),
        )
        st.caption(strategy_gate.message)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Check": check.name,
                        "Status": "pass" if check.passed else "blocked",
                        "Detail": check.detail,
                    }
                    for check in strategy_gate.checks
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

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
