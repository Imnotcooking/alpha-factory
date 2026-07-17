"""Unified operations cockpit for Alpha Factory."""

from __future__ import annotations

import json
import os
import re
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
    account_performance_summary,
    account_positions_display,
    account_top_positions,
    account_trade_event_summary,
    account_trade_events_display,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_trade_events,
    load_latest_account_nav,
    load_latest_account_positions,
)
from oqp.config import load_settings  # noqa: E402
from oqp.intelligence import EngineContext, default_intelligence_coordinator  # noqa: E402
from oqp.ops import collect_ops_status  # noqa: E402
from oqp.paper_trading import (  # noqa: E402
    PaperOrderTicketStatus,
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_nav,
    load_latest_paper_orders,
    load_latest_paper_positions,
    paper_order_notional_today,
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
from oqp.ui import (  # noqa: E402
    apply_ops_theme,
    language_selector,
    nav_tiles,
    ops_text,
    page_header,
    qmt_account_rows,
    qmt_overall_status,
    qmt_safety_gate_frame,
    qmt_status_frame,
    qmt_submit_state,
    render_dark_bar_chart,
    render_dark_line_chart,
    render_dark_table,
    render_market_lane_overview,
    section_header,
    tr,
)


st.set_page_config(
    page_title="Ops Dashboard",
    layout="wide",
    page_icon="OPS",
    initial_sidebar_state="expanded",
)

OPS_LANG = language_selector()
apply_ops_theme()


def T(key: str, default: str | None = None, **format_values: Any) -> str:
    return ops_text(OPS_LANG, key, default, **format_values)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def money(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return "missing"
    return f"{number:,.2f}"


def signed_money(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:,.2f}"


def compact_money(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return "missing"
    sign = "-" if number < 0 else ""
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M"
    if absolute >= 10_000:
        return f"{sign}{absolute / 1_000:.1f}k"
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.2f}k"
    return f"{number:,.2f}"


def compact_signed_money(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    return f"{sign}{compact_money(number)}"


def percent(value: Any) -> str:
    number = coerce_float(value)
    if number is None:
        return "missing"
    return f"{number * 100:.2f}%"


def human_timestamp(value: Any) -> str:
    if not value:
        return "missing"
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %d %H:%M UTC")


ISO_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)


def humanize_detail_timestamps(value: Any) -> str:
    text = str(value or "")

    def replace(match: re.Match[str]) -> str:
        return human_timestamp(match.group(0))

    return ISO_TIMESTAMP_RE.sub(replace, text)


def source_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "account ledger" in text:
        return "Ledger"
    if "portfolio ledger" in text:
        return "Portfolio"
    return str(value or "missing")


def status_label(status: str) -> str:
    return status.upper()


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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


def status_dataframe(rows: list[dict[str, object]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    render_dark_table(frame, empty_message="No status rows available.", max_height_px=420)


def engine_results_frame(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for result in results.values():
        rows.append(
            {
                "Engine": result.engine_name,
                "Engine ID": result.engine_id,
                "Status": status_label(result.status.value),
                "Summary": result.summary,
                "Generated": result.generated_at.isoformat(timespec="seconds"),
                "Frames": ", ".join(result.frames) if result.frames else "none",
            }
        )
    return pd.DataFrame(rows)


def render_engine_result(result: Any) -> None:
    st.subheader(result.engine_name)
    cols = st.columns(4)
    cols[0].metric("Status", status_label(result.status.value))
    cols[1].metric("Engine ID", result.engine_id)
    cols[2].metric("Frames", str(len(result.frames)))
    cols[3].metric("Generated", result.generated_at.isoformat(timespec="seconds"))

    if result.status.value == "pass":
        st.success(result.summary)
    elif result.status.value in {"warn", "skipped"}:
        st.warning(result.summary)
    else:
        st.error(result.summary)

    if result.metrics:
        render_dark_table(
            pd.DataFrame(
                [
                    {"Metric": key, "Value": value}
                    for key, value in result.metrics.items()
                ]
            ),
        )

    if result.frames:
        frame_tabs = st.tabs([name.replace("_", " ").title() for name in result.frames])
        for tab, (name, frame) in zip(frame_tabs, result.frames.items()):
            with tab:
                if frame.empty:
                    st.info(f"No rows available for {name}.")
                else:
                    render_dark_table(frame, max_height_px=420)

    if result.signals:
        st.caption(f"Signals: {json.dumps(result.signals, default=str)}")


def category_rows(frame: pd.DataFrame, category: str) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return frame[frame["Category"].eq(category)].to_dict("records")


STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


def worst_status(rows: pd.DataFrame) -> str:
    if rows.empty or "Status" not in rows:
        return "pass"
    statuses = [str(status) for status in rows["Status"].dropna().tolist()]
    if not statuses:
        return "pass"
    return max(statuses, key=lambda status: STATUS_RANK.get(status, 1))


def check_subset(
    frame: pd.DataFrame,
    *,
    categories: tuple[str, ...] = (),
    contains: tuple[str, ...] = (),
) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = pd.Series(False, index=frame.index)
    if categories:
        mask = mask | frame["Category"].isin(categories)
    for text in contains:
        mask = mask | frame["Check"].str.contains(text, case=False, na=False)
    return frame[mask]


def status_detail(rows: pd.DataFrame, fallback: str) -> str:
    if rows.empty:
        return fallback
    failed = rows[rows["Status"].eq("fail")]
    warned = rows[rows["Status"].eq("warn")]
    source = failed if not failed.empty else warned
    if source.empty:
        return fallback
    first = source.iloc[0]
    return humanize_detail_timestamps(f"{first.get('Check')}: {first.get('Detail')}")


def latest_nav_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[0]


def environment_summary(
    *,
    environment: str,
    nav_frame: pd.DataFrame,
    nav_history_raw: pd.DataFrame,
    positions: pd.DataFrame,
    fallback_nav: pd.DataFrame | None = None,
) -> dict[str, Any]:
    row = latest_nav_row(nav_frame)
    fallback_row = None if fallback_nav is None or fallback_nav.empty else fallback_nav.iloc[0]
    nav = None if row is None else coerce_float(row.get("net_liquidation"))
    cash = None if row is None else coerce_float(row.get("cash"))
    daily_pnl = None if row is None else coerce_float(row.get("daily_pnl"))
    position_count = None if row is None else int(coerce_float(row.get("position_count")) or 0)
    as_of = "" if row is None else str(row.get("as_of") or "")
    source = "unified account ledger" if row is not None else "missing"

    if row is None and fallback_row is not None:
        nav = coerce_float(
            fallback_row.get("net_liquidation")
            if "net_liquidation" in fallback_row
            else fallback_row.get("total_net_worth")
        )
        cash = coerce_float(
            fallback_row.get("cash")
            if "cash" in fallback_row
            else fallback_row.get("total_cash")
        )
        daily_pnl = coerce_float(fallback_row.get("daily_pnl"))
        position_count = int(coerce_float(fallback_row.get("position_count")) or 0)
        as_of = str(fallback_row.get("as_of") or fallback_row.get("date") or "")
        source = "legacy paper ledger" if environment == "paper" else "legacy portfolio ledger"

    performance = account_performance_summary(
        nav_history_raw,
        positions,
        current_nav=nav,
        current_cash=cash,
        current_daily_pnl=daily_pnl,
    )
    return {
        "environment": environment,
        "nav": nav,
        "cash": cash,
        "daily_pnl": daily_pnl,
        "position_count": len(positions) if position_count is None else position_count,
        "as_of": as_of,
        "source": source,
        "performance": performance,
    }


def summary_metric_row(name: str, summary: dict[str, Any]) -> dict[str, str]:
    performance = summary["performance"]
    return {
        "Account": name,
        "NAV": compact_money(summary["nav"]),
        "Total Cash": compact_money(summary["cash"]),
        "P&L": compact_signed_money(summary["daily_pnl"]),
        "Pos": str(summary["position_count"]),
        "Day %": percent(performance.get("daily_return")),
        "Max Drawdown": percent(performance.get("max_drawdown_pct")),
        "Gross Exposure / NAV": percent(performance.get("gross_exposure_pct")),
        "Source": source_label(summary["source"]),
        "Updated": human_timestamp(summary["as_of"]),
    }


def account_signal(summary: dict[str, Any]) -> str:
    return (
        f"NAV {money(summary['nav'])}; cash {money(summary['cash'])}; "
        f"P&L {signed_money(summary['daily_pnl'])}; positions {summary['position_count']}"
    )


def status_count_frame(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty or "Status" not in items:
        return pd.DataFrame(columns=["Status", "Checks"])
    return (
        items["Status"]
        .astype(str)
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Checks")
    )


def account_state_rows(name: str, summary: dict[str, Any]) -> list[dict[str, str]]:
    performance = summary["performance"]
    return [
        {"Metric": "Account", "Value": name},
        {"Metric": "NAV", "Value": money(summary["nav"])},
        {"Metric": "Total Cash", "Value": money(summary["cash"])},
        {"Metric": "Daily P&L", "Value": signed_money(summary["daily_pnl"])},
        {"Metric": "Daily Return", "Value": percent(performance.get("daily_return"))},
        {
            "Metric": "Cumulative Return",
            "Value": percent(performance.get("cumulative_return")),
        },
        {"Metric": "Max Drawdown", "Value": percent(performance.get("max_drawdown_pct"))},
        {"Metric": "Cash Weight", "Value": percent(performance.get("cash_pct"))},
        {"Metric": "Gross Exposure / NAV", "Value": percent(performance.get("gross_exposure_pct"))},
        {"Metric": "Source", "Value": str(summary["source"])},
        {"Metric": "As Of", "Value": human_timestamp(summary["as_of"])},
    ]


def position_concentration_frame(
    positions: pd.DataFrame,
    *,
    nav: float | None,
    limit: int = 10,
) -> pd.DataFrame:
    top = account_top_positions(positions, limit=limit)
    columns = [
        "Symbol",
        "Asset Class",
        "Market Value",
        "NAV Weight",
        "Unrealized P&L",
        "Realized P&L",
    ]
    if top.empty:
        return pd.DataFrame(columns=columns)

    out = top.copy()
    out["nav_weight"] = (
        0.0
        if nav in (None, 0)
        else pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0) / float(nav)
    )
    return (
        out.rename(
            columns={
                "symbol": "Symbol",
                "asset_class": "Asset Class",
                "market_value": "Market Value",
                "nav_weight": "NAV Weight",
                "unrealized_pnl": "Unrealized P&L",
                "realized_pnl": "Realized P&L",
            }
        )
        .reindex(columns=columns)
    )


def combined_concentration_frame(
    *,
    live_positions: pd.DataFrame,
    paper_positions: pd.DataFrame,
    live_nav: float | None,
    paper_nav: float | None,
) -> pd.DataFrame:
    live = position_concentration_frame(live_positions, nav=live_nav).assign(
        Account="Live"
    )
    paper = position_concentration_frame(paper_positions, nav=paper_nav).assign(
        Account="Paper"
    )
    combined = pd.concat([live, paper], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(
            columns=[
                "Account",
                "Symbol",
                "Asset Class",
                "Market Value",
                "NAV Weight",
                "Unrealized P&L",
                "Realized P&L",
            ]
        )
    return combined[
        [
            "Account",
            "Symbol",
            "Asset Class",
            "Market Value",
            "NAV Weight",
            "Unrealized P&L",
            "Realized P&L",
        ]
    ]


def action_queue_rows(
    *,
    items: pd.DataFrame,
    live_summary: dict[str, Any],
    paper_summary: dict[str, Any],
    paper_review_ready: bool,
    paper_submit_ready: bool,
    dry_run_count: int,
    approved_count: int,
    blocked_review_count: int,
    ready_review_count: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(priority: str, area: str, state: str, detail: str, next_action: str) -> None:
        rows.append(
            {
                "Priority": priority,
                "Area": area,
                "State": state,
                "Detail": humanize_detail_timestamps(detail),
                "Next Action": next_action,
            }
        )

    if not items.empty:
        for _, row in items[items["Status"].eq("fail")].head(5).iterrows():
            add(
                "high",
                str(row.get("Category", "")),
                "fail",
                f"{row.get('Check')}: {row.get('Detail')}",
                "inspect the System tab",
            )
        for _, row in items[items["Status"].eq("warn")].head(5).iterrows():
            add(
                "medium",
                str(row.get("Category", "")),
                "warn",
                f"{row.get('Check')}: {row.get('Detail')}",
                "inspect the relevant detail page",
            )

    if dry_run_count:
        add(
            "high",
            "Execution",
            "waiting",
            f"{dry_run_count} dry-run ticket(s) need a human decision",
            "open Execution and approve or reject",
        )
    if approved_count and not paper_submit_ready:
        add(
            "high",
            "Paper Submit",
            "locked",
            f"{approved_count} approved ticket(s) cannot be submitted yet",
            "keep simulation-only or intentionally arm paper submit",
        )
    elif approved_count and paper_submit_ready:
        add(
            "high",
            "Paper Submit",
            "ready",
            f"{approved_count} approved ticket(s) are eligible for submit preflight",
            "run submitter after final review",
        )
    if blocked_review_count:
        add(
            "medium",
            "Safety Review",
            "blocked",
            f"{blocked_review_count} proposal review(s) blocked",
            "inspect review messages in Execution",
        )
    if ready_review_count:
        add(
            "low",
            "Safety Review",
            "ready",
            f"{ready_review_count} proposal review(s) passed",
            "watch ticket creation and human decision",
        )

    for label, summary in (("Live", live_summary), ("Paper", paper_summary)):
        if summary.get("nav") is None:
            add(
                "medium",
                f"{label} Account",
                "missing",
                "NAV is not available",
                f"check {label.lower()} snapshot job and account ledger",
            )

    if not paper_review_ready:
        add(
            "low",
            "Paper Review",
            "locked",
            "paper review gate is not open",
            "leave locked unless actively reviewing proposals",
        )

    if not rows:
        add(
            "none",
            "Dashboard",
            "clear",
            "no immediate action detected",
            "monitor next scheduled snapshot and strategy cycle",
        )
    return rows


def ticket_status_frame(paper_orders: pd.DataFrame) -> pd.DataFrame:
    if paper_orders.empty or "status" not in paper_orders:
        return pd.DataFrame(columns=["Status", "Tickets", "Meaning"])
    counts = paper_orders["status"].astype(str).value_counts().to_dict()
    rows = [
        {
            "Status": PaperOrderTicketStatus.DRY_RUN.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.DRY_RUN.value, 0)),
            "Meaning": "waiting for human approve/reject",
        },
        {
            "Status": PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value,
            "Tickets": int(
                counts.get(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value, 0)
            ),
            "Meaning": "approved but still needs submit gate/preflight",
        },
        {
            "Status": PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value,
            "Tickets": int(
                counts.get(PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value, 0)
            ),
            "Meaning": "submitted to IBKR paper; monitor fills/events",
        },
        {
            "Status": PaperOrderTicketStatus.REJECTED.value,
            "Tickets": int(counts.get(PaperOrderTicketStatus.REJECTED.value, 0)),
            "Meaning": "closed by human decision",
        },
    ]
    other_count = sum(
        int(value)
        for key, value in counts.items()
        if key not in {row["Status"] for row in rows}
    )
    if other_count:
        rows.append(
            {"Status": "other", "Tickets": other_count, "Meaning": "unexpected status"}
        )
    return pd.DataFrame(rows)


def risk_limit_rows(
    *,
    live_summary: dict[str, Any],
    paper_summary: dict[str, Any],
    paper_daily_notional_used: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, summary in (("Live", live_summary), ("Paper", paper_summary)):
        performance = summary["performance"]
        gross_pct = coerce_float(performance.get("gross_exposure_pct"))
        cash_pct = coerce_float(performance.get("cash_pct"))
        drawdown = coerce_float(performance.get("max_drawdown_pct"))
        rows.extend(
            [
                {
                    "Account": label,
                    "Risk Check": "Gross exposure / NAV",
                    "Value": percent(gross_pct),
                    "Read": "higher means more balance-sheet use",
                },
                {
                    "Account": label,
                    "Risk Check": "Cash weight",
                    "Value": percent(cash_pct),
                    "Read": "liquidity buffer",
                },
                {
                    "Account": label,
                    "Risk Check": "Max drawdown",
                    "Value": percent(drawdown),
                    "Read": "worst recorded NAV drawdown",
                },
            ]
        )
    rows.append(
        {
            "Account": "Paper",
            "Risk Check": "Daily notional used",
            "Value": money(paper_daily_notional_used),
            "Read": "paper tickets created today",
        }
    )
    return rows


def policy_gate_rows(
    *,
    settings: Any,
    paper_daily_notional_used: float,
    paper_review_ready: bool,
    paper_submit_ready: bool,
) -> list[dict[str, str]]:
    return [
        {
            "Policy": "Trading mode",
            "State": settings.trading_mode,
            "Limit / Setting": "runtime profile",
            "Detail": "Selects the broker profile used by execution tooling.",
        },
        {
            "Policy": "Live trading",
            "State": "open" if settings.allow_live_trading else "locked",
            "Limit / Setting": f"ALLOW_LIVE_TRADING={str(settings.allow_live_trading).lower()}",
            "Detail": "Must remain locked until live deployment is intentional.",
        },
        {
            "Policy": "Live monitor",
            "State": "enabled" if settings.ibkr_live_monitor_enabled else "disabled",
            "Limit / Setting": (
                "IBKR_LIVE_MONITOR_ENABLED="
                + str(settings.ibkr_live_monitor_enabled).lower()
            ),
            "Detail": "Allows read-only live account visibility.",
        },
        {
            "Policy": "Paper review",
            "State": "active" if paper_review_ready else "locked",
            "Limit / Setting": f"ALLOW_PAPER_TRADING={str(settings.allow_paper_trading).lower()}",
            "Detail": "Controls whether paper proposals can pass safety review.",
        },
        {
            "Policy": "Paper broker submit",
            "State": "armed" if paper_submit_ready else "locked",
            "Limit / Setting": (
                "ALLOW_PAPER_ORDER_SUBMIT="
                + str(settings.allow_paper_order_submit).lower()
            ),
            "Detail": "Separate final gate before tickets can reach IBKR paper.",
        },
        {
            "Policy": "Paper order cap",
            "State": "configured" if settings.paper_max_order_notional is not None else "missing",
            "Limit / Setting": money(settings.paper_max_order_notional),
            "Detail": "Maximum notional for one reviewed paper order.",
        },
        {
            "Policy": "Paper daily cap",
            "State": "configured" if settings.paper_max_daily_notional is not None else "missing",
            "Limit / Setting": (
                f"{money(paper_daily_notional_used)} used / "
                f"{money(settings.paper_max_daily_notional)} max"
            ),
            "Detail": "Daily notional budget consumed by paper tickets.",
        },
        {
            "Policy": "Paper asset classes",
            "State": "configured" if settings.paper_allowed_asset_classes else "open",
            "Limit / Setting": ", ".join(settings.paper_allowed_asset_classes) or "not restricted",
            "Detail": "Asset classes accepted by the paper safety reviewer.",
        },
        {
            "Policy": "Paper options",
            "State": "enabled" if settings.paper_options_enabled else "locked",
            "Limit / Setting": f"PAPER_OPTIONS_ENABLED={str(settings.paper_options_enabled).lower()}",
            "Detail": "Option proposals stay blocked unless this policy is enabled.",
        },
    ]


def system_summary_rows(items: pd.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for category in [
        "Gateway",
        "Broker Heartbeat",
        "Accounts",
        "Jobs",
        "Schedulers",
        "Notifications",
        "Safety",
        "Host",
    ]:
        subset = category_rows(items, category)
        frame = pd.DataFrame(subset)
        rows.append(
            {
                "Area": category,
                "Status": status_label(worst_status(frame)),
                "Checks": str(len(frame)),
                "Detail": status_detail(frame, "all checks passing"),
            }
        )
    return rows


def pipeline_rows(
    items: pd.DataFrame,
    *,
    live_summary: dict[str, Any],
    paper_summary: dict[str, Any],
    paper_review_ready: bool,
    paper_submit_ready: bool,
    demo_mode: bool = False,
) -> list[dict[str, str]]:
    paper = check_subset(items, contains=("Paper",))
    live = check_subset(items, contains=("Live", "Portfolio Snapshot"))
    safety = check_subset(items, categories=("Safety",))
    ops = check_subset(items, categories=("Schedulers", "Host", "Notifications", "Jobs"))
    gateway = check_subset(items, categories=("Gateway", "Broker Heartbeat"))
    qmt = check_subset(items, contains=("QMT",))

    return [
        {
            "Layer": "Research",
            "Mode": "Separate dashboard",
            "Status": "LOCAL",
            "Signal": "Factor work, backtests, and candidate promotion stay in research.",
        },
        {
            "Layer": "Live Portfolio",
            "Mode": "Read-only",
            "Status": status_label(worst_status(live)),
            "Signal": status_detail(live, account_signal(live_summary)),
        },
        {
            "Layer": "Paper Trading",
            "Mode": "Paper account",
            "Status": status_label(worst_status(paper)),
            "Signal": status_detail(paper, account_signal(paper_summary)),
        },
        {
            "Layer": "Execution",
            "Mode": "Review" if paper_review_ready else "Locked",
            "Status": status_label(worst_status(safety)),
            "Signal": (
                "paper review open; paper submit "
                + ("armed" if paper_submit_ready else "locked")
            ),
        },
        {
            "Layer": "QMT Connector",
            "Mode": "Not connected" if demo_mode else "Read-only / guarded submit",
            "Status": "DEMO" if demo_mode else status_label(worst_status(qmt)),
            "Signal": (
                "Optional adapter; no QMT process is contacted in demo mode."
                if demo_mode
                else status_detail(qmt, "QMT bridge is installed; connector state follows runtime flags.")
            ),
        },
        {
            "Layer": "IBKR And Server",
            "Mode": "Not connected" if demo_mode else "Monitoring",
            "Status": "DEMO" if demo_mode else status_label(worst_status(gateway)),
            "Signal": (
                "Optional adapter; no IBKR gateway or server is contacted in demo mode."
                if demo_mode
                else status_detail(gateway, "IBKR sockets and API heartbeats are stable.")
            ),
        },
        {
            "Layer": "Jobs And Alerts",
            "Mode": "Not scheduled" if demo_mode else "Automated",
            "Status": "DEMO" if demo_mode else status_label(worst_status(ops)),
            "Signal": (
                "Schedulers and notifications are intentionally inactive in demo mode."
                if demo_mode
                else status_detail(ops, "Snapshot timers, host, and Discord checks are stable.")
            ),
        },
    ]


def ticket_display(frame: pd.DataFrame) -> pd.DataFrame:
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
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        metadata = parse_metadata(row.get("metadata_json"))
        rows.append(
            {
                "Created": row.get("created_at"),
                "Status": row.get("status"),
                "Symbol": row.get("symbol"),
                "Side": row.get("side"),
                "Quantity": row.get("quantity"),
                "Type": row.get("order_type"),
                "Limit": row.get("limit_price"),
                "Strategy": row.get("strategy_id") or "",
                "Proposal": metadata.get("proposal_id", ""),
                "Broker Submit": yes_no(bool(metadata.get("broker_submit_enabled"))),
                "Ticket": row.get("order_id"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def review_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Reviewed",
        "Decision",
        "Proposal",
        "Orders",
        "Estimated Notional",
        "Message",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    display = frame.copy()
    display["Estimated Notional"] = display["estimated_notional"].map(money)
    return (
        display.rename(
            columns={
                "reviewed_at": "Reviewed",
                "decision": "Decision",
                "proposal_id": "Proposal",
                "order_count": "Orders",
                "message": "Message",
            }
        )
        .reindex(columns=columns)
    )


def legacy_live_position_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "broker",
        "ticker",
        "asset_type",
        "shares",
        "avg_cost",
        "current_price",
        "unrealized_pnl",
        "currency",
    ]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Broker",
                "Ticker",
                "Asset Type",
                "Shares",
                "Avg Cost",
                "Current Price",
                "Unrealized P&L",
                "Currency",
            ]
        )
    return frame.reindex(columns=columns).rename(
        columns={
            "date": "Date",
            "broker": "Broker",
            "ticker": "Ticker",
            "asset_type": "Asset Type",
            "shares": "Shares",
            "avg_cost": "Avg Cost",
            "current_price": "Current Price",
            "unrealized_pnl": "Unrealized P&L",
            "currency": "Currency",
        }
    )


def render_account_section(
    *,
    name: str,
    summary: dict[str, Any],
    nav_history: pd.DataFrame,
    positions: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    metrics = st.columns(6)
    metrics[0].metric("NAV", money(summary["nav"]))
    metrics[1].metric("Total Cash", money(summary["cash"]))
    metrics[2].metric("Daily P&L", signed_money(summary["daily_pnl"]))
    metrics[3].metric("Positions", str(summary["position_count"]))
    metrics[4].metric("Max Drawdown", percent(summary["performance"].get("max_drawdown_pct")))
    metrics[5].metric("Source", str(summary["source"]))

    overview, holdings, events_tab, state = st.tabs(
        ["Overview", "Holdings", "Events", "State"]
    )

    with overview:
        chart_left, chart_right = st.columns([1.2, 1])
        with chart_left:
            st.subheader(f"{name} Equity")
            if nav_history.empty:
                st.info("No NAV history has been recorded yet.")
            else:
                render_dark_line_chart(
                    nav_history.set_index("date")[["net_liquidation", "equity_peak"]],
                    yaxis_title="Account Value",
                )
        with chart_right:
            st.subheader(f"{name} Drawdown")
            if nav_history.empty:
                st.info("No drawdown history has been recorded yet.")
            else:
                render_dark_line_chart(
                    nav_history.set_index("date")[["drawdown_pct"]],
                    yaxis_title="Drawdown",
                )

        st.subheader(f"{name} Account State")
        render_dark_table(
            pd.DataFrame(account_state_rows(name, summary)),
        )

    with holdings:
        position_left, position_right = st.columns([1.2, 1])
        with position_left:
            st.subheader(f"{name} Positions")
            position_table = account_positions_display(positions)
            if position_table.empty:
                st.info("No position rows are available.")
            else:
                render_dark_table(position_table, max_height_px=420)
        with position_right:
            st.subheader(f"{name} Asset Mix")
            mix = account_asset_summary(positions)
            if mix.empty:
                st.info("No asset mix is available.")
            else:
                render_dark_table(mix, max_height_px=320)
                render_dark_bar_chart(
                    mix.set_index("Asset Class")[["Market Value"]],
                    yaxis_title="Market Value",
                )

        st.subheader(f"{name} Concentration")
        concentration = position_concentration_frame(
            positions,
            nav=coerce_float(summary.get("nav")),
        )
        if concentration.empty:
            st.info("No concentration rows are available.")
        else:
            render_dark_table(concentration, max_height_px=420)

    with events_tab:
        st.subheader(f"{name} Events")
        if events.empty:
            st.info("No account events have been recorded yet.")
        else:
            event_left, event_right = st.columns([1.3, 1])
            with event_left:
                render_dark_table(
                    account_trade_events_display(events),
                    max_height_px=420,
                )
            with event_right:
                render_dark_table(
                    account_trade_event_summary(events),
                    max_height_px=420,
                )

    with state:
        st.subheader(f"{name} Source Contract")
        render_dark_table(
            pd.DataFrame(
                [
                    {"Field": "Environment", "Value": str(summary["environment"])},
                    {"Field": "Source", "Value": str(summary["source"])},
                    {"Field": "As Of", "Value": human_timestamp(summary["as_of"])},
                    {
                        "Field": "NAV Observations",
                        "Value": str(summary["performance"].get("nav_observations")),
                    },
                    {"Field": "Position Rows", "Value": str(len(positions))},
                    {"Field": "Event Rows", "Value": str(len(events))},
                ]
            ),
        )


settings = load_settings()
status_kwargs: dict[str, Path] = {}
if os.getenv("OQP_PROFILE") == "demo":
    demo_log_root = Path(os.environ.get("OQP_RUNTIME_ROOT", REPO_ROOT / "runtime" / "demo")) / "logs"
    status_kwargs = {
        "portfolio_health_path": demo_log_root / "portfolio_snapshot_health.json",
        "paper_health_path": demo_log_root / "paper_trading_health.json",
        "ibkr_heartbeat_health_path": demo_log_root / "ibkr_adapter_heartbeat_health.json",
        "server_ibkr_readiness_path": demo_log_root / "server_ibkr_readiness_health.json",
    }
snapshot = collect_ops_status(
    settings=settings,
    demo_mode=os.getenv("OQP_PROFILE") == "demo",
    **status_kwargs,
)
items_df = pd.DataFrame(snapshot.item_rows)
account_status_df = pd.DataFrame(snapshot.account_rows)
event_status_df = pd.DataFrame(snapshot.event_rows)

account_ledger_path = default_account_ledger_path()
paper_ledger_path = default_paper_trading_ledger_path()
portfolio_ledger_path = default_portfolio_ledger_path()

live_nav = load_latest_account_nav(account_ledger_path, environment="live")
paper_nav = load_latest_account_nav(account_ledger_path, environment="paper")
live_nav_raw_history = load_account_nav_history(account_ledger_path, environment="live")
paper_nav_raw_history = load_account_nav_history(account_ledger_path, environment="paper")
live_nav_history = account_nav_drawdowns(live_nav_raw_history)
paper_nav_history = account_nav_drawdowns(paper_nav_raw_history)
live_positions = load_latest_account_positions(account_ledger_path, environment="live")
paper_positions = load_latest_account_positions(account_ledger_path, environment="paper")
live_events = load_account_trade_events(account_ledger_path, environment="live", limit=25)
paper_events = load_account_trade_events(account_ledger_path, environment="paper", limit=25)

legacy_paper_nav = load_latest_paper_nav(paper_ledger_path)
legacy_paper_positions = load_latest_paper_positions(paper_ledger_path)
paper_orders = load_latest_paper_orders(paper_ledger_path, limit=50)
paper_reviews = load_latest_paper_execution_reviews(paper_ledger_path, limit=25)
paper_daily_notional_used = paper_order_notional_today(paper_ledger_path)

legacy_live_nav_history = compute_nav_drawdowns(load_historical_nav(portfolio_ledger_path))
legacy_live_positions = load_latest_live_positions(portfolio_ledger_path)
legacy_live_nav = (
    pd.DataFrame()
    if legacy_live_nav_history.empty
    else legacy_live_nav_history.tail(1).rename(
        columns={
            "total_net_worth": "net_liquidation",
            "total_cash": "cash",
        }
    )
)

live_summary = environment_summary(
    environment="live",
    nav_frame=live_nav,
    nav_history_raw=live_nav_raw_history,
    positions=live_positions,
    fallback_nav=legacy_live_nav,
)
paper_summary = environment_summary(
    environment="paper",
    nav_frame=paper_nav,
    nav_history_raw=paper_nav_raw_history,
    positions=paper_positions,
    fallback_nav=legacy_paper_nav,
)

overall_counts = items_df["Status"].value_counts().to_dict() if not items_df.empty else {}
failed_or_warn = (
    items_df[items_df["Status"].isin(["fail", "warn"])] if not items_df.empty else pd.DataFrame()
)
paper_review_ready = bool(
    settings.trading_mode.lower() == "paper"
    and not settings.allow_live_trading
    and settings.allow_paper_trading
)
paper_submit_ready = bool(settings.allow_paper_order_submit)
dry_run_count = (
    0
    if paper_orders.empty or "status" not in paper_orders
    else int(paper_orders["status"].eq(PaperOrderTicketStatus.DRY_RUN.value).sum())
)
approved_count = (
    0
    if paper_orders.empty or "status" not in paper_orders
    else int(
        paper_orders["status"].eq(PaperOrderTicketStatus.APPROVED_FOR_SUBMIT.value).sum()
    )
)
submitted_count = (
    0
    if paper_orders.empty or "status" not in paper_orders
    else int(
        paper_orders["status"].eq(PaperOrderTicketStatus.SUBMITTED_TO_BROKER.value).sum()
    )
)
blocked_review_count = (
    0
    if paper_reviews.empty or "decision" not in paper_reviews
    else int(paper_reviews["decision"].eq("blocked").sum())
)
ready_review_count = (
    0
    if paper_reviews.empty or "decision" not in paper_reviews
    else int(paper_reviews["decision"].eq("ready").sum())
)
ibkr_metrics = read_json(DEFAULT_IBKR_METRICS_PATH)
manual_inputs = read_json(DEFAULT_PORTFOLIO_MANUAL_INPUTS_PATH)
banked_profits = read_json(DEFAULT_BANKED_PROFITS_PATH)
active_runtime_root = Path(os.environ.get("OQP_RUNTIME_ROOT", REPO_ROOT / "runtime"))
server_sync = read_json(active_runtime_root / "state" / "server_sync" / "status.json")
action_queue = pd.DataFrame(
    action_queue_rows(
        items=items_df,
        live_summary=live_summary,
        paper_summary=paper_summary,
        paper_review_ready=paper_review_ready,
        paper_submit_ready=paper_submit_ready,
        dry_run_count=dry_run_count,
        approved_count=approved_count,
        blocked_review_count=blocked_review_count,
        ready_review_count=ready_review_count,
    )
)
actionable_count = int(
    action_queue[~action_queue["Priority"].astype(str).eq("none")].shape[0]
    if not action_queue.empty
    else 0
)
ticket_status = ticket_status_frame(paper_orders)
system_summary = pd.DataFrame(system_summary_rows(items_df))
page_header(
    title="Ops Dashboard",
    title_zh="运营驾驶舱",
    subtitle="Unified cockpit for live monitoring, paper trading, execution safety, and server health.",
    subtitle_zh="统一监控实盘、模拟交易、执行安全与服务器健康状态。",
    language=OPS_LANG,
)
render_market_lane_overview(language=OPS_LANG, expanded=True)
st.caption(T("checked_at", time=human_timestamp(snapshot.checked_at)))
if server_sync:
    st.caption(
        T(
            "server_runtime_sync",
            synced_at=human_timestamp(server_sync.get("synced_at")),
            remote=server_sync.get("remote", "server"),
        )
    )

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Command Status", "指挥状态"),
        subtitle=tr(
            OPS_LANG,
            "Top-line gates, account readiness, and current operating state.",
            "核心闸门、账户就绪状态与当前运行状态。",
        ),
        accent="teal",
    )
    top_cols = st.columns(10)
    top_cols[0].metric(T("overall"), status_label(snapshot.overall_status))
    top_cols[1].metric(T("failures"), str(overall_counts.get("fail", 0)))
    top_cols[2].metric(T("warnings"), str(overall_counts.get("warn", 0)))
    top_cols[3].metric(T("live_nav"), money(live_summary["nav"]))
    top_cols[4].metric(T("paper_nav"), money(paper_summary["nav"]))
    top_cols[5].metric(T("dry_run_tickets"), str(dry_run_count))
    top_cols[6].metric(T("paper_submit"), T("armed") if paper_submit_ready else T("locked"))
    top_cols[7].metric(T("action_items"), str(actionable_count))
    top_cols[8].metric("QMT Heartbeat", qmt_overall_status(snapshot))
    top_cols[9].metric("QMT Submit", qmt_submit_state(settings))

    if paper_submit_ready:
        st.warning(T("paper_submission_armed"))
    elif settings.allow_live_trading:
        st.error(T("live_trading_enabled"))
    else:
        st.info(T("live_locked"))

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Workspaces", "工作区"),
        subtitle=tr(
            OPS_LANG,
            "Jump into the daily operating rooms without leaving the Ops cockpit.",
            "从运营驾驶舱快速进入每日工作页面。",
        ),
        accent="blue",
    )
    nav_tiles(
        [
            (
                tr(OPS_LANG, "Live Portfolio", "实盘组合"),
                "/Live_Portfolio",
                tr(OPS_LANG, "Real money monitor", "实盘监控"),
            ),
            (
                tr(OPS_LANG, "Paper Trading", "模拟交易"),
                "/Paper_Trading",
                tr(OPS_LANG, "Paper account and trials", "模拟账户与试运行"),
            ),
            (
                tr(OPS_LANG, "Discretionary", "主观工作台"),
                "/Discretionary_Workbench",
                tr(OPS_LANG, "Valuation and options tools", "估值与期权工具"),
            ),
            (
                tr(OPS_LANG, "Risk Control", "风险控制"),
                "/Risk_Control_Room",
                tr(OPS_LANG, "Exposure and allocation", "敞口与配置"),
            ),
            (
                tr(OPS_LANG, "Execution", "执行监控"),
                "/Execution_Strategy_Monitor",
                tr(OPS_LANG, "Orders and strategy ops", "订单与策略运营"),
            ),
            (
                tr(OPS_LANG, "Journal", "日志报告"),
                "/Journal_Reports",
                tr(OPS_LANG, "Notes and daily reports", "复盘与日报"),
            ),
        ]
    )

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Action Queue", "行动队列"),
        subtitle=tr(
            OPS_LANG,
            "The highest-priority warnings and failures from gateways, accounts, jobs, and safety checks.",
            "来自网关、账户、任务与安全检查的最高优先级事项。",
        ),
        accent="rose",
    )
    render_dark_table(action_queue, empty_message=T("no_action_rows"), max_height_px=380)
    if actionable_count == 0:
        st.success(T("no_immediate_action"))

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Operating Pipeline", "运营流水线"),
        subtitle=tr(
            OPS_LANG,
            "How research, accounts, execution, broker gateways, jobs, and alerts currently line up.",
            "研究、账户、执行、券商网关、任务与通知当前如何衔接。",
        ),
        accent="amber",
    )
    render_dark_table(
        pd.DataFrame(
            pipeline_rows(
                items_df,
                live_summary=live_summary,
                paper_summary=paper_summary,
                paper_review_ready=paper_review_ready,
                paper_submit_ready=paper_submit_ready,
                demo_mode=os.getenv("OQP_PROFILE") == "demo",
            )
        ),
        empty_message=T("no_pipeline_rows"),
    )

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Limits & Policy", "限制与政策"),
        subtitle=tr(
            OPS_LANG,
            "Runtime guardrails that decide whether paper or live actions are allowed.",
            "决定模拟或实盘动作是否允许的运行时护栏。",
        ),
        accent="green",
    )
    policy_cols = st.columns(4)
    policy_cols[0].metric(
        tr(OPS_LANG, "Live Trading", "实盘交易"),
        T("open") if settings.allow_live_trading else T("locked"),
    )
    policy_cols[1].metric(
        tr(OPS_LANG, "Paper Review", "模拟复核"),
        T("active") if paper_review_ready else T("locked"),
    )
    policy_cols[2].metric(T("paper_submit"), T("armed") if paper_submit_ready else T("locked"))
    policy_cols[3].metric(
        tr(OPS_LANG, "Paper Daily Used", "模拟日度已用"),
        money(paper_daily_notional_used),
    )
    render_dark_table(
        pd.DataFrame(
            policy_gate_rows(
                settings=settings,
                paper_daily_notional_used=paper_daily_notional_used,
                paper_review_ready=paper_review_ready,
                paper_submit_ready=paper_submit_ready,
            )
        ),
        empty_message=T("no_policy_rows"),
    )
    st.subheader("QMT Bridge Gates")
    render_dark_table(qmt_safety_gate_frame(settings), max_height_px=280)

account_left, account_right = st.columns([1.2, 1])
with account_left:
    with st.container(border=True):
        section_header(
            tr(OPS_LANG, "Account Summary", "账户摘要"),
            subtitle=tr(
                OPS_LANG,
                "Live and paper account state from the unified account ledger.",
                "来自统一账户账本的实盘与模拟账户状态。",
            ),
            accent="blue",
        )
        render_dark_table(
            pd.DataFrame(
                [
                    summary_metric_row("Live", live_summary),
                    summary_metric_row("Paper", paper_summary),
                ]
            ),
            empty_message=T("no_account_summary_rows"),
        )
        st.subheader("QMT Account Rows")
        render_dark_table(
            qmt_account_rows(snapshot),
            empty_message="No QMT account rows are available yet.",
            max_height_px=240,
        )
with account_right:
    with st.container(border=True):
        section_header(
            tr(OPS_LANG, "System Summary", "系统摘要"),
            subtitle=tr(
                OPS_LANG,
                "Gateway, scheduler, notification, and host checks.",
                "网关、调度、通知与主机检查。",
            ),
            accent="teal",
        )
        render_dark_table(system_summary, empty_message=T("no_system_summary_rows"))
        st.subheader("QMT System Rows")
        render_dark_table(
            qmt_status_frame(snapshot),
            empty_message="No QMT status rows are available yet.",
            max_height_px=240,
        )

with st.container(border=True):
    section_header(
        tr(OPS_LANG, "Attention", "需要关注"),
        subtitle=tr(
            OPS_LANG,
            "Full warning and failure detail for anything that still needs review.",
            "仍需查看的警告与失败细节。",
        ),
        accent="rose",
    )
    if failed_or_warn.empty:
        st.success(T("all_ops_passing"))
    else:
        status_dataframe(failed_or_warn)

st.caption(T("unified_account_ledger", path=display_path(account_ledger_path)))
st.stop()
