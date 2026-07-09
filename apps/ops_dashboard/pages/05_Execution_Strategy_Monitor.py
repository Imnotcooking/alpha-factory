"""Execution and graduated strategy monitor for the Ops dashboard."""

from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    account_nav_drawdowns,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_trade_events,
)
from oqp.config import load_settings  # noqa: E402
from oqp.execution.artifacts import (  # noqa: E402
    load_trade_proposal_artifacts,
    trade_proposal_directory,
)
from oqp.intelligence import (  # noqa: E402
    EngineContext,
    PortfolioManagerEngine,
    RegimeSnapshotEngine,
)
from oqp.ops import collect_ops_status  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    PaperOrderTicketStatus,
    default_paper_trading_ledger_path,
    ensure_paper_trading_schema,
    load_latest_paper_execution_reviews,
    load_latest_paper_orders,
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
    qmt_overall_status,
    qmt_route_candidate_frame,
    qmt_strategy_route_frame,
    qmt_submit_state,
    render_dark_bar_chart,
    render_dark_table,
    render_market_lane_chips,
    render_qmt_audit_panel,
    render_qmt_connector_panel,
    render_qmt_safety_panel,
)


st.set_page_config(
    page_title="Execution & Strategy Monitor",
    layout="wide",
    page_icon="EXEC",
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


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
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
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def status_label(status: str) -> str:
    return str(status).upper()


@st.cache_data(ttl=60)
def load_monitor_data() -> dict[str, Any]:
    settings = load_settings()
    account_ledger = default_account_ledger_path()
    paper_ledger = default_paper_trading_ledger_path()
    proposal_load = load_trade_proposal_artifacts(
        trade_proposal_directory(settings),
        max_files=100,
    )
    paper_nav_raw = load_account_nav_history(account_ledger, environment="paper")
    live_nav_raw = load_account_nav_history(account_ledger, environment="live")
    paper_events = load_account_trade_events(
        account_ledger,
        environment="paper",
        limit=250,
    )
    orders = load_latest_paper_orders(paper_ledger, limit=250)
    reviews = load_latest_paper_execution_reviews(paper_ledger, limit=250)
    registry = load_paper_strategy_registry(paper_ledger, limit=500)
    fills = load_paper_fills(paper_ledger, limit=250)
    ops_snapshot = collect_ops_status(settings=settings)
    return {
        "settings": settings,
        "account_ledger": account_ledger,
        "paper_ledger": paper_ledger,
        "proposal_load": proposal_load,
        "paper_nav_raw": paper_nav_raw,
        "live_nav_raw": live_nav_raw,
        "paper_events": paper_events,
        "orders": orders,
        "reviews": reviews,
        "registry": registry,
        "fills": fills,
        "ops_snapshot": ops_snapshot,
        "daily_notional_used": paper_order_notional_today(paper_ledger),
    }


def load_paper_fills(db_path: str | Path, *, limit: int = 100) -> pd.DataFrame:
    columns = [
        "fill_id",
        "order_id",
        "executed_at",
        "strategy_id",
        "symbol",
        "side",
        "quantity",
        "average_price",
        "commission",
        "currency",
        "metadata_json",
    ]
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    ensure_paper_trading_schema(path)
    with closing(sqlite3.connect(path)) as conn:
        return pd.read_sql(
            """
            SELECT fill_id, order_id, executed_at, strategy_id, symbol, side, quantity,
                   average_price, commission, currency, metadata_json
            FROM paper_fills
            ORDER BY executed_at DESC, fill_id DESC
            LIMIT ?
            """,
            conn,
            params=(max(int(limit), 1),),
        )


def proposal_queue_frame(load_result: Any) -> pd.DataFrame:
    columns = [
        "Created",
        "Proposal",
        "Status",
        "Strategy",
        "Source",
        "Paper Only",
        "Intents",
        "Symbols",
        "Estimated Notional",
        "Artifact",
    ]
    rows: list[dict[str, Any]] = []
    for loaded in load_result.loaded:
        proposal = loaded.proposal
        symbols = sorted({intent.instrument.symbol for intent in proposal.intents})
        rows.append(
            {
                "Created": proposal.created_at.isoformat(timespec="seconds"),
                "Proposal": proposal.proposal_id,
                "Status": proposal.status.value,
                "Strategy": proposal.strategy_id or _strategy_from_intents(proposal.intents),
                "Source": proposal.source,
                "Paper Only": yes_no(proposal.paper_only),
                "Intents": len(proposal.intents),
                "Symbols": ", ".join(symbols),
                "Estimated Notional": proposal.estimated_notional,
                "Artifact": display_path(loaded.path),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def proposal_intents_frame(load_result: Any) -> pd.DataFrame:
    columns = [
        "Proposal",
        "Strategy",
        "Symbol",
        "Asset Class",
        "Side",
        "Quantity",
        "Order Type",
        "Limit",
        "Reference",
        "Target Weight",
        "Confidence",
        "Signal",
        "Rationale",
    ]
    rows: list[dict[str, Any]] = []
    for loaded in load_result.loaded:
        proposal = loaded.proposal
        for intent in proposal.intents:
            rows.append(
                {
                    "Proposal": proposal.proposal_id,
                    "Strategy": intent.strategy_id or proposal.strategy_id or "",
                    "Symbol": intent.instrument.symbol,
                    "Asset Class": intent.instrument.asset_class.value,
                    "Side": intent.side.value,
                    "Quantity": intent.quantity,
                    "Order Type": intent.order_type.value,
                    "Limit": intent.limit_price,
                    "Reference": intent.reference_price,
                    "Target Weight": intent.target_weight,
                    "Confidence": intent.confidence,
                    "Signal": intent.signal_id or "",
                    "Rationale": intent.rationale or "",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def proposal_issues_frame(load_result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Artifact": display_path(issue.path), "Issue": issue.message}
            for issue in load_result.issues
        ],
        columns=["Artifact", "Issue"],
    )


def _strategy_from_intents(intents: tuple[Any, ...]) -> str:
    ids = sorted({str(intent.strategy_id) for intent in intents if intent.strategy_id})
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        return ", ".join(ids)
    return ""


def signal_frame(proposals: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy_id",
        "active",
        "strength",
        "latest_proposal",
        "latest_status",
        "latest_symbols",
        "latest_created",
    ]
    if proposals.empty:
        return pd.DataFrame(columns=columns)

    out = proposals.copy()
    out["strategy_id"] = out["Strategy"].replace("", pd.NA)
    out = out.dropna(subset=["strategy_id"])
    if out.empty:
        return pd.DataFrame(columns=columns)

    if not intents.empty and "Confidence" in intents:
        confidence = (
            intents.dropna(subset=["Strategy"])
            .assign(Confidence=lambda frame: pd.to_numeric(frame["Confidence"], errors="coerce"))
            .groupby("Strategy")["Confidence"]
            .max()
            .to_dict()
        )
    else:
        confidence = {}

    latest = out.sort_values("Created").groupby("strategy_id").tail(1)
    rows = []
    for _, row in latest.iterrows():
        status = str(row.get("Status") or "")
        rows.append(
            {
                "strategy_id": row["strategy_id"],
                "active": status in {"draft", "approved"},
                "strength": confidence.get(row["strategy_id"], 1.0 if status == "approved" else 0.5),
                "latest_proposal": row.get("Proposal"),
                "latest_status": status,
                "latest_symbols": row.get("Symbols"),
                "latest_created": row.get("Created"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
        "Source",
        "Notes",
    ]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in registry.to_dict("records"):
        rows.append(
            {
                "Strategy": row.get("strategy_id"),
                "Market": row.get("market_vertical"),
                "Candidate": row.get("candidate_id"),
                "Status": row.get("status"),
                "Kill Switch": yes_no(row.get("kill_switch")),
                "Max Order": money(row.get("max_order_notional")),
                "Max Daily": money(row.get("max_daily_notional")),
                "Symbols": ", ".join(parse_json_list(row.get("allowed_symbols_json"))),
                "Approved": row.get("approved_at"),
                "Source": row.get("source"),
                "Notes": row.get("notes") or "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def ticket_display(orders: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Created",
        "Status",
        "Strategy",
        "Symbol",
        "Side",
        "Quantity",
        "Type",
        "Limit",
        "Proposal",
        "Review",
        "Broker Submit",
        "Ticket",
    ]
    if orders.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in orders.to_dict("records"):
        metadata = parse_json_dict(row.get("metadata_json"))
        rows.append(
            {
                "Created": row.get("created_at"),
                "Status": row.get("status"),
                "Strategy": row.get("strategy_id") or "",
                "Symbol": row.get("symbol"),
                "Side": row.get("side"),
                "Quantity": row.get("quantity"),
                "Type": row.get("order_type"),
                "Limit": row.get("limit_price"),
                "Proposal": metadata.get("proposal_id", ""),
                "Review": metadata.get("review_id", ""),
                "Broker Submit": yes_no(metadata.get("broker_submit_enabled")),
                "Ticket": row.get("order_id"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def review_display(reviews: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Reviewed",
        "Decision",
        "Proposal",
        "Orders",
        "Estimated Notional",
        "Failed Checks",
        "Message",
    ]
    if reviews.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in reviews.to_dict("records"):
        rows.append(
            {
                "Reviewed": row.get("reviewed_at"),
                "Decision": row.get("decision"),
                "Proposal": row.get("proposal_id"),
                "Orders": row.get("order_count"),
                "Estimated Notional": row.get("estimated_notional"),
                "Failed Checks": ", ".join(failed_review_checks(row.get("checks_json"))),
                "Message": row.get("message") or "",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def failed_review_checks(value: Any) -> list[str]:
    try:
        checks = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(checks, list):
        return []
    return [
        str(check.get("name") or "unknown")
        for check in checks
        if isinstance(check, dict) and not bool(check.get("passed"))
    ]


def order_status_counts(orders: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    statuses = [
        PaperOrderTicketStatus.DRY_RUN.value,
        PaperOrderTicketStatus.BLOCKED.value,
        PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
        PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value,
        "filled",
        "cancelled",
        PaperOrderTicketStatus.REJECTED.value,
    ]
    counts = orders["status"].astype(str).value_counts().to_dict() if "status" in orders else {}
    rows = [
        {"State": status, "Rows": int(counts.get(status, 0))}
        for status in statuses
    ]
    if not fills.empty:
        rows[4]["Rows"] = len(fills)
    return pd.DataFrame(rows)


def review_status_counts(reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty or "decision" not in reviews:
        return pd.DataFrame(columns=["Decision", "Reviews", "Estimated Notional"])
    out = reviews.copy()
    out["estimated_notional"] = pd.to_numeric(out["estimated_notional"], errors="coerce").fillna(0.0)
    return (
        out.groupby("decision", dropna=False)
        .agg(Reviews=("proposal_id", "count"), **{"Estimated Notional": ("estimated_notional", "sum")})
        .reset_index()
        .rename(columns={"decision": "Decision"})
    )


def strategy_activity_frame(
    *,
    registry: pd.DataFrame,
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
    fills: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Registry Status",
        "Kill Switch",
        "Orders",
        "Dry Run",
        "Approved",
        "Submitted",
        "Rejected",
        "Reviews Ready",
        "Reviews Blocked",
        "Fills",
        "Gross Fill Value",
        "Recorded Strategy P&L",
        "P&L Source",
    ]
    ids: set[str] = set()
    for frame in (registry, orders, fills, events):
        if not frame.empty and "strategy_id" in frame:
            ids.update(str(value) for value in frame["strategy_id"].dropna() if str(value).strip())
    if not registry.empty and "strategy_id" in registry:
        ids.update(str(value) for value in registry["strategy_id"].dropna() if str(value).strip())
    if not ids:
        return pd.DataFrame(columns=columns)

    registry_lookup = {
        str(row.get("strategy_id")): row
        for row in registry.to_dict("records")
        if row.get("strategy_id")
    }
    rows = []
    for strategy_id in sorted(ids):
        strategy_orders = _strategy_filter(orders, strategy_id)
        strategy_reviews = _reviews_for_strategy(reviews, strategy_orders)
        strategy_fills = _strategy_filter(fills, strategy_id)
        strategy_events = _strategy_filter(events, strategy_id)
        statuses = (
            strategy_orders["status"].astype(str).value_counts().to_dict()
            if not strategy_orders.empty and "status" in strategy_orders
            else {}
        )
        pnl, pnl_source = strategy_pnl(strategy_events, strategy_fills)
        record = registry_lookup.get(strategy_id, {})
        rows.append(
            {
                "Strategy": strategy_id,
                "Registry Status": record.get("status", "unregistered"),
                "Kill Switch": yes_no(record.get("kill_switch")),
                "Orders": len(strategy_orders),
                "Dry Run": int(statuses.get(PaperOrderTicketStatus.DRY_RUN.value, 0)),
                "Approved": int(statuses.get(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value, 0)),
                "Submitted": int(statuses.get(PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value, 0)),
                "Rejected": int(statuses.get(PaperOrderTicketStatus.REJECTED.value, 0)),
                "Reviews Ready": _review_decision_count(strategy_reviews, "ready"),
                "Reviews Blocked": _review_decision_count(strategy_reviews, "blocked"),
                "Fills": len(strategy_fills),
                "Gross Fill Value": gross_fill_value(strategy_fills),
                "Recorded Strategy P&L": pnl,
                "P&L Source": pnl_source,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _strategy_filter(frame: pd.DataFrame, strategy_id: str) -> pd.DataFrame:
    if frame.empty or "strategy_id" not in frame:
        return pd.DataFrame()
    return frame[frame["strategy_id"].astype(str).eq(strategy_id)].copy()


def _reviews_for_strategy(reviews: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty or orders.empty:
        return pd.DataFrame()
    proposal_ids = set()
    for value in orders.get("metadata_json", pd.Series(dtype=object)):
        metadata = parse_json_dict(value)
        proposal_id = metadata.get("proposal_id")
        if proposal_id:
            proposal_ids.add(str(proposal_id))
    if not proposal_ids or "proposal_id" not in reviews:
        return pd.DataFrame()
    return reviews[reviews["proposal_id"].astype(str).isin(proposal_ids)].copy()


def _review_decision_count(reviews: pd.DataFrame, decision: str) -> int:
    if reviews.empty or "decision" not in reviews:
        return 0
    return int(reviews["decision"].astype(str).eq(decision).sum())


def gross_fill_value(fills: pd.DataFrame) -> float | None:
    if fills.empty:
        return None
    qty = pd.to_numeric(fills.get("quantity"), errors="coerce").fillna(0.0).abs()
    price = pd.to_numeric(fills.get("average_price"), errors="coerce").fillna(0.0)
    return float((qty * price).sum())


def strategy_pnl(events: pd.DataFrame, fills: pd.DataFrame) -> tuple[float | None, str]:
    values = []
    for frame in (events, fills):
        if frame.empty or "metadata_json" not in frame:
            continue
        for raw in frame["metadata_json"].tolist():
            metadata = parse_json_dict(raw)
            for key in ("realized_pnl", "unrealized_pnl", "pnl", "pnl_delta"):
                value = _float(metadata.get(key))
                if value is not None:
                    values.append(value)
    if values:
        return float(sum(values)), "event/fill metadata"
    return None, "pending attribution"


def allocation_frame(activity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Strategy",
        "Recent Gross Fill Value",
        "Recent Activity Weight",
        "Recorded P&L",
        "Allocation Read",
    ]
    if activity.empty:
        return pd.DataFrame(columns=columns)
    out = activity.copy()
    out["Recent Gross Fill Value"] = pd.to_numeric(out["Gross Fill Value"], errors="coerce").fillna(0.0)
    total = float(out["Recent Gross Fill Value"].abs().sum())
    out["Recent Activity Weight"] = (
        0.0
        if total == 0
        else (out["Recent Gross Fill Value"].abs() / total) * 100.0
    )
    out["Recorded P&L"] = pd.to_numeric(out["Recorded Strategy P&L"], errors="coerce")
    out["Allocation Read"] = out["Recent Activity Weight"].map(
        lambda value: "waiting for fills" if value == 0 else "activity-derived"
    )
    return out.reindex(columns=columns)


def regime_compatibility_frame(registry: pd.DataFrame, regime_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["Strategy", "Market", "Current Paper Regime", "Declared Compatible Regimes", "Compatibility"]
    if registry.empty:
        return pd.DataFrame(columns=columns)
    paper_regime = "unknown"
    if not regime_frame.empty and "Account" in regime_frame and "State" in regime_frame:
        paper_rows = regime_frame[regime_frame["Account"].astype(str).str.lower().eq("paper")]
        if not paper_rows.empty:
            paper_regime = str(paper_rows.iloc[0].get("State") or "unknown")
    rows = []
    for row in registry.to_dict("records"):
        metadata = parse_json_dict(row.get("metadata_json"))
        declared = metadata.get("compatible_regimes") or metadata.get("regime_compatibility") or []
        if isinstance(declared, str):
            declared_list = [declared]
        elif isinstance(declared, list):
            declared_list = [str(item) for item in declared]
        else:
            declared_list = []
        if not declared_list:
            compatibility = "not declared"
        elif paper_regime in declared_list:
            compatibility = "compatible"
        else:
            compatibility = "mismatch"
        rows.append(
            {
                "Strategy": row.get("strategy_id"),
                "Market": row.get("market_vertical"),
                "Current Paper Regime": paper_regime,
                "Declared Compatible Regimes": ", ".join(declared_list) or "not declared",
                "Compatibility": compatibility,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def live_boundary_rows(settings: Any, registry: pd.DataFrame, daily_notional_used: float) -> pd.DataFrame:
    running = _registry_status_count(registry, "paper_running")
    paused = _registry_status_count(registry, "paused")
    retired = _registry_status_count(registry, "retired")
    kill_switches = int(registry["kill_switch"].fillna(0).astype(bool).sum()) if "kill_switch" in registry else 0
    return pd.DataFrame(
        [
            {
                "Gate": "Live trading",
                "State": "open" if settings.allow_live_trading else "locked",
                "Detail": f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
            },
            {
                "Gate": "Paper review",
                "State": "active" if settings.allow_paper_trading else "locked",
                "Detail": f"ALLOW_PAPER_TRADING={str(settings.allow_paper_trading).lower()}",
            },
            {
                "Gate": "Paper submit",
                "State": "armed" if settings.allow_paper_order_submit else "locked",
                "Detail": f"ALLOW_PAPER_ORDER_SUBMIT={str(settings.allow_paper_order_submit).lower()}",
            },
            {
                "Gate": "QMT paper submit",
                "State": "armed" if settings.allow_qmt_paper_order_submit else "locked",
                "Detail": f"ALLOW_QMT_PAPER_ORDER_SUBMIT={str(settings.allow_qmt_paper_order_submit).lower()}",
            },
            {
                "Gate": "QMT live submit",
                "State": "armed" if settings.allow_qmt_live_trading else "locked",
                "Detail": f"ALLOW_QMT_LIVE_TRADING={str(settings.allow_qmt_live_trading).lower()}",
            },
            {
                "Gate": "QMT submit connector",
                "State": "isolated"
                if settings.qmt_submit_connector_url.rstrip("/") != settings.qmt_connector_url.rstrip("/")
                else "shared",
                "Detail": settings.qmt_submit_connector_url,
            },
            {
                "Gate": "Daily paper notional",
                "State": "tracking",
                "Detail": f"{money(daily_notional_used)} used / {money(settings.paper_max_daily_notional)} max",
            },
            {
                "Gate": "Strategy kill switches",
                "State": "clear" if kill_switches == 0 else "active",
                "Detail": f"{kill_switches} active; running={running}, paused={paused}, retired={retired}",
            },
            {
                "Gate": "Live eligibility",
                "State": "future",
                "Detail": "This page monitors paper graduates; live eligibility stays locked until later governance.",
            },
        ]
    )


def _registry_status_count(registry: pd.DataFrame, status: str) -> int:
    if registry.empty or "status" not in registry:
        return 0
    return int(registry["status"].astype(str).eq(status).sum())


def notifications_frame(snapshot: Any) -> pd.DataFrame:
    frame = pd.DataFrame(snapshot.item_rows)
    columns = ["Category", "Check", "Status", "Detail"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    mask = frame["Category"].astype(str).isin(["Notifications", "Safety", "Schedulers", "Jobs"])
    return frame[mask].reindex(columns=columns)


def summary_table(label: str, value: Any, detail: str) -> dict[str, Any]:
    return {"Metric": label, "Value": value, "Detail": detail}


data = load_monitor_data()
settings = data["settings"]
paper_nav = account_nav_drawdowns(data["paper_nav_raw"])
live_nav = account_nav_drawdowns(data["live_nav_raw"])
proposals = proposal_queue_frame(data["proposal_load"])
proposal_intents = proposal_intents_frame(data["proposal_load"])
proposal_issues = proposal_issues_frame(data["proposal_load"])
signals = signal_frame(proposals, proposal_intents)
orders = data["orders"]
reviews = data["reviews"]
registry = data["registry"]
fills = data["fills"]
events = data["paper_events"]
ticket_counts = order_status_counts(orders, fills)
review_counts = review_status_counts(reviews)
activity = strategy_activity_frame(
    registry=registry,
    orders=orders,
    reviews=reviews,
    fills=fills,
    events=events,
)
allocation = allocation_frame(activity)
context = EngineContext(
    environment="ops",
    settings={
        "trading_mode": settings.trading_mode,
        "allow_paper_trading": settings.allow_paper_trading,
        "allow_live_trading": settings.allow_live_trading,
        "allow_paper_order_submit": settings.allow_paper_order_submit,
    },
    paper_nav_history=paper_nav,
    live_nav_history=live_nav,
    paper_events=events,
    approved_strategies=registry,
    strategy_signals=signals,
    metadata={"source": "execution_strategy_monitor"},
)
pm_result = PortfolioManagerEngine().run(context)
regime_result = RegimeSnapshotEngine().run(context)
runtime_decisions = pm_result.frame("runtime_decisions")
regime_frame = regime_result.frame("regimes")
regime_compatibility = regime_compatibility_frame(registry, regime_frame)
boundary = live_boundary_rows(settings, registry, data["daily_notional_used"])
notifications = notifications_frame(data["ops_snapshot"])


page_header(
    title="Execution & Strategy Monitor",
    title_zh="执行与策略监控",
    subtitle="One operations page for graduated strategies, proposal flow, order states, safety reviews, and execution boundaries.",
    subtitle_zh="集中监控已毕业策略、交易提案流、订单状态、安全审查与执行边界。",
    language=OPS_LANG,
)
render_market_lane_chips(
    language=OPS_LANG,
    lanes=("EQUITY_US", "OPTIONS_US", "EQUITY_CN", "OPTIONS_CN", "FUTURES_CN"),
    caption="Strategy graduation and execution review must stay market-scoped before a proposal can become a paper/live ticket.",
)

ready_reviews = _review_decision_count(reviews, "ready")
blocked_reviews = _review_decision_count(reviews, "blocked")
dry_run = int(
    orders["status"].astype(str).eq(PaperOrderTicketStatus.DRY_RUN.value).sum()
    if not orders.empty and "status" in orders
    else 0
)
approved = int(
    orders["status"].astype(str).eq(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value).sum()
    if not orders.empty and "status" in orders
    else 0
)
submitted = int(
    orders["status"].astype(str).eq(PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value).sum()
    if not orders.empty and "status" in orders
    else 0
)
running_strategies = _registry_status_count(registry, "paper_running")
kill_switches = int(registry["kill_switch"].fillna(0).astype(bool).sum()) if "kill_switch" in registry else 0

top = st.columns(10)
top[0].metric(T("registered", "Registered"), str(len(registry)))
top[1].metric(T("running", "Running"), str(running_strategies))
top[2].metric(T("kill_switches", "Kill Switches"), str(kill_switches))
top[3].metric(T("proposals", "Proposals"), str(len(proposals)))
top[4].metric(T("ready_reviews", "Ready Reviews"), str(ready_reviews))
top[5].metric(T("blocked_reviews", "Blocked Reviews"), str(blocked_reviews))
top[6].metric(T("dry_run_tickets"), str(dry_run))
top[7].metric(T("submit_gate"), T("armed") if settings.allow_paper_order_submit else T("locked"))
top[8].metric("QMT Heartbeat", qmt_overall_status(data["ops_snapshot"]))
top[9].metric("QMT Submit", qmt_submit_state(settings))

if settings.allow_live_trading:
    st.error(T("execution_live_enabled", "Live trading is enabled. Treat this as a production boundary until reviewed."))
elif settings.allow_paper_order_submit:
    st.warning(T("execution_paper_armed", "Paper submit is armed. Approved tickets may be eligible for guarded paper broker submission."))
else:
    st.info(T("execution_locked", "Live trading is locked and paper submit is locked. This page is monitoring proposals, tickets, and strategy state."))

command_tab, strategy_tab, orders_tab, signals_tab, boundary_tab = st.tabs(
    ops_tabs(OPS_LANG, "execution_tabs")
)

with command_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Execution State")
        render_dark_table(ticket_counts, max_height_px=280)
        if not ticket_counts.empty:
            render_dark_bar_chart(ticket_counts.set_index("State")[["Rows"]])
    with right:
        st.subheader("Safety Reviews")
        if review_counts.empty:
            st.info("No safety reviews have been recorded yet.")
        else:
            render_dark_table(review_counts, max_height_px=280)
            render_dark_bar_chart(review_counts.set_index("Decision")[["Reviews"]])

    st.subheader("Trade Proposal Queue")
    if proposals.empty:
        st.info("No trade proposal artifacts are currently waiting in the runtime artifact directory.")
    else:
        render_dark_table(proposals, max_height_px=460)

    if not proposal_issues.empty:
        st.subheader("Proposal Artifact Issues")
        render_dark_table(proposal_issues, max_height_px=360)

    st.subheader("Current Operating Read")
    render_dark_table(
        pd.DataFrame(
            [
                summary_table("Portfolio Manager", status_label(pm_result.status.value), pm_result.summary),
                summary_table("Regime Snapshot", status_label(regime_result.status.value), regime_result.summary),
                summary_table("Daily Paper Notional", money(data["daily_notional_used"]), "Paper ticket notional consumed today."),
                summary_table("Paper Submit", "armed" if settings.allow_paper_order_submit else "locked", "Final broker submission gate."),
                summary_table("QMT Submit", qmt_submit_state(settings), "Separate connector and HMAC gate for QMT routes."),
            ]
        ),
    )

    st.subheader("QMT Command Read")
    render_qmt_connector_panel(settings, data["ops_snapshot"], compact=True)
    render_qmt_audit_panel(settings, limit=8)

with strategy_tab:
    st.subheader("Graduated Strategy Roster")
    roster = registry_display(registry)
    if roster.empty:
        st.info("No strategy has graduated into the paper strategy registry yet.")
    else:
        status_left, roster_right = st.columns([1, 2])
        with status_left:
            render_dark_table(
                registry["status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Strategies"),
                max_height_px=280,
            )
        with roster_right:
            render_dark_table(roster, max_height_px=460)

    st.subheader("Strategy Activity & P&L Attribution")
    if activity.empty:
        st.info("No strategy-level activity has been recorded yet.")
    else:
        render_dark_table(
            activity,
            max_height_px=460,
        )
        if activity["Recorded Strategy P&L"].notna().any():
            pnl_chart = activity.dropna(subset=["Recorded Strategy P&L"]).set_index("Strategy")[["Recorded Strategy P&L"]]
            render_dark_bar_chart(pnl_chart)
        else:
            st.caption("Strategy P&L attribution will populate once fills/events carry strategy-level P&L metadata.")

    st.subheader("Regime Compatibility")
    if regime_compatibility.empty:
        st.info("No registered strategy has regime compatibility metadata yet.")
    else:
        render_dark_table(regime_compatibility, max_height_px=420)

    st.subheader("QMT Strategy Route Eligibility")
    render_dark_table(
        qmt_strategy_route_frame(registry),
        empty_message="No strategies are registered for QMT route review yet.",
        max_height_px=360,
    )

with orders_tab:
    order_left, review_right = st.columns([1.15, 1])
    with order_left:
        st.subheader("Latest Paper Tickets")
        tickets = ticket_display(orders)
        if tickets.empty:
            st.info("No paper tickets are available.")
        else:
            render_dark_table(tickets, max_height_px=460)
    with review_right:
        st.subheader("Latest Safety Reviews")
        display_reviews = review_display(reviews)
        if display_reviews.empty:
            st.info("No paper execution reviews are available.")
        else:
            render_dark_table(display_reviews, max_height_px=460)

    fill_left, event_right = st.columns([1, 1])
    with fill_left:
        st.subheader("Paper Fills")
        if fills.empty:
            st.info("No paper fills have been recorded yet.")
        else:
            render_dark_table(fills, max_height_px=420)
    with event_right:
        st.subheader("Paper Execution Events")
        if events.empty:
            st.info("No paper account execution events have been recorded yet.")
        else:
            render_dark_table(
                events.reindex(
                    columns=[
                        "occurred_at",
                        "event_type",
                        "strategy_id",
                        "symbol",
                        "side",
                        "quantity",
                        "price",
                        "order_id",
                    ]
                ),
                max_height_px=420,
            )

    st.subheader("QMT Order Audit")
    render_qmt_audit_panel(settings, limit=15)

with signals_tab:
    st.subheader("Latest Operational Signals")
    if signals.empty:
        st.info("No strategy-linked proposal signals are available yet.")
    else:
        render_dark_table(signals, max_height_px=420)

    st.subheader("Portfolio Manager Runtime Decisions")
    if runtime_decisions.empty:
        requirements = pm_result.frame("requirements")
        if requirements.empty:
            st.info(pm_result.summary)
        else:
            render_dark_table(requirements, max_height_px=420)
    else:
        render_dark_table(runtime_decisions, max_height_px=420)

    st.subheader("Current Allocation Read")
    if allocation.empty:
        st.info("No strategy allocation read is available yet.")
    else:
        render_dark_table(
            allocation,
            max_height_px=420,
        )
        if allocation["Recent Activity Weight"].sum() > 0:
            render_dark_bar_chart(allocation.set_index("Strategy")[["Recent Activity Weight"]])

    st.subheader("Proposal Intents")
    if proposal_intents.empty:
        st.info("No order intents are available from proposal artifacts.")
    else:
        render_dark_table(proposal_intents, max_height_px=420)

    st.subheader("QMT Route Candidates")
    if proposal_intents.empty or "Symbol" not in proposal_intents.columns:
        render_dark_table(qmt_strategy_route_frame(registry), empty_message="No QMT route candidates are available yet.")
    else:
        symbols = proposal_intents["Symbol"].dropna().astype(str).head(12).tolist()
        if not symbols:
            st.info("No proposal symbols are available for QMT route classification.")
        else:
            route_rows = pd.concat(
                [qmt_route_candidate_frame(symbol, settings).assign(Symbol=symbol) for symbol in symbols],
                ignore_index=True,
            )
            render_dark_table(route_rows, max_height_px=360)

with boundary_tab:
    boundary_left, alert_right = st.columns([1.1, 1])
    with boundary_left:
        st.subheader("Live Boundary & Kill Switches")
        render_dark_table(boundary, max_height_px=420)
    with alert_right:
        st.subheader("Discord & Job Alerts")
        if notifications.empty:
            st.info("No notification or scheduler checks are available.")
        else:
            render_dark_table(notifications, max_height_px=420)

    st.subheader("Regime Snapshot")
    if regime_frame.empty:
        st.info("No regime snapshot is available.")
    else:
        render_dark_table(
            regime_frame,
            max_height_px=420,
        )

    st.subheader("Ledger Contracts")
    render_dark_table(
        pd.DataFrame(
            [
                {"Contract": "Proposal artifacts", "Path": display_path(trade_proposal_directory(settings))},
                {"Contract": "Paper trading ledger", "Path": display_path(data["paper_ledger"])},
                {"Contract": "Unified account ledger", "Path": display_path(data["account_ledger"])},
            ]
        ),
    )

    st.subheader("QMT Boundary Gates")
    render_qmt_safety_panel(settings, max_height_px=360)

    st.subheader("QMT Connector Contracts")
    render_dark_table(qmt_connector_contract_frame(settings), max_height_px=260)
