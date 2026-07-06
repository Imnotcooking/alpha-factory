"""Operator journal, trade thesis log, and report archive."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    default_account_ledger_path,
    load_account_nav_history,
    load_account_trade_events,
)
from oqp.journal import (  # noqa: E402
    default_journal_ledger_path,
    ensure_journal_schema,
    load_journal_entries,
    write_journal_entry,
)
from oqp.paper_trading import (  # noqa: E402
    default_paper_trading_ledger_path,
    load_latest_paper_execution_reviews,
    load_latest_paper_orders,
)
from oqp.ui import apply_ops_theme, language_selector, ops_tabs, ops_text, page_header, render_dark_table  # noqa: E402


st.set_page_config(
    page_title="Journal & Reports",
    layout="wide",
    page_icon="JOURNAL",
    initial_sidebar_state="expanded",
)

OPS_LANG = language_selector()
apply_ops_theme()


def T(key: str, default: str | None = None, **format_values: Any) -> str:
    return ops_text(OPS_LANG, key, default, **format_values)


REPORT_SCAN_ROOTS = (
    REPO_ROOT / "reports",
    REPO_ROOT / "runtime" / "exports",
    REPO_ROOT / "runtime" / "artifacts",
    REPO_ROOT / "logs",
)


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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def csv_list(value: str, *, uppercase: bool = False) -> list[str]:
    values = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        values.append(text.upper() if uppercase else text)
    return values


def json_list_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if not value:
        return ""
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return ""
    if not isinstance(decoded, list):
        return ""
    return ", ".join(str(item) for item in decoded)


def json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def filter_by_date(frame: pd.DataFrame, column: str, selected: date) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return pd.DataFrame(columns=frame.columns)
    parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame[parsed.dt.date.eq(selected)].copy()


def latest_nav_for_date(frame: pd.DataFrame, selected: date) -> pd.Series | None:
    if frame.empty or "date" not in frame:
        return None
    rows = frame[frame["date"].astype(str).str[:10].eq(selected.isoformat())]
    if rows.empty:
        return None
    return rows.sort_values("as_of").iloc[-1]


def nav_metric_rows(live_nav: pd.DataFrame, paper_nav: pd.DataFrame, selected: date) -> pd.DataFrame:
    rows = []
    for label, frame in (("Live", live_nav), ("Paper", paper_nav)):
        current = latest_nav_for_date(frame, selected)
        previous_rows = frame[frame["date"].astype(str).str[:10].lt(selected.isoformat())] if "date" in frame else pd.DataFrame()
        previous = previous_rows.sort_values("date").iloc[-1] if not previous_rows.empty else None
        nav = None if current is None else _float(current.get("net_liquidation"))
        prior_nav = None if previous is None else _float(previous.get("net_liquidation"))
        daily_pnl = None if current is None else _float(current.get("daily_pnl"))
        if daily_pnl in (None, 0.0) and nav is not None and prior_nav is not None:
            daily_pnl = nav - prior_nav
        rows.append(
            {
                "Account": label,
                "NAV": nav,
                "Cash": None if current is None else _float(current.get("cash")),
                "Daily P&L": daily_pnl,
                "Positions": None if current is None else current.get("position_count"),
                "As Of": "missing" if current is None else str(current.get("as_of") or ""),
            }
        )
    return pd.DataFrame(rows)


def digest_text(
    *,
    selected: date,
    nav_rows: pd.DataFrame,
    live_events: pd.DataFrame,
    paper_events: pd.DataFrame,
    paper_orders: pd.DataFrame,
    paper_reviews: pd.DataFrame,
    journal_entries: pd.DataFrame,
) -> str:
    lines = [f"EOD summary for {selected.isoformat()}"]
    if nav_rows.empty:
        lines.append("- NAV: no live or paper NAV rows available.")
    else:
        for row in nav_rows.to_dict("records"):
            lines.append(
                "- "
                + f"{row['Account']}: NAV {money(row.get('NAV'))}, "
                + f"cash {money(row.get('Cash'))}, "
                + f"daily P&L {signed_money(row.get('Daily P&L'))}, "
                + f"positions {row.get('Positions') if row.get('Positions') is not None else 'missing'}."
            )
    lines.extend(
        [
            f"- Live events: {len(live_events)}.",
            f"- Paper events: {len(paper_events)}.",
            f"- Paper tickets created: {len(paper_orders)}.",
            f"- Paper safety reviews: {len(paper_reviews)}.",
            f"- Journal entries saved today: {len(journal_entries)}.",
        ]
    )
    if not paper_reviews.empty and "decision" in paper_reviews:
        counts = paper_reviews["decision"].astype(str).value_counts().to_dict()
        lines.append("- Review decisions: " + ", ".join(f"{key}={value}" for key, value in counts.items()) + ".")
    if not paper_orders.empty and "status" in paper_orders:
        counts = paper_orders["status"].astype(str).value_counts().to_dict()
        lines.append("- Ticket states: " + ", ".join(f"{key}={value}" for key, value in counts.items()) + ".")
    return "\n".join(lines)


def journal_display(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date",
        "Category",
        "Environment",
        "Title",
        "Symbols",
        "Strategies",
        "Tags",
        "Body",
        "Mistake",
        "Lesson",
        "Follow Up",
        "Created",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    return pd.DataFrame(
        {
            "Date": out["entry_date"],
            "Category": out["category"],
            "Environment": out["environment"].fillna(""),
            "Title": out["title"],
            "Symbols": out["symbols_json"].map(json_list_text),
            "Strategies": out["strategies_json"].map(json_list_text),
            "Tags": out["tags_json"].map(json_list_text),
            "Body": out["body"].fillna(""),
            "Mistake": out["mistake"].fillna(""),
            "Lesson": out["lesson"].fillna(""),
            "Follow Up": out["follow_up"].fillna(""),
            "Created": out["created_at"],
        }
    )


def thesis_display(frame: pd.DataFrame) -> pd.DataFrame:
    thesis = frame[frame["category"].eq("trade_thesis")].copy() if not frame.empty else pd.DataFrame()
    columns = [
        "Date",
        "Symbol",
        "Strategy",
        "Status",
        "Direction",
        "Horizon",
        "Confidence",
        "Title",
        "Thesis",
        "Invalidation",
        "Expected Outcome",
    ]
    if thesis.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in thesis.to_dict("records"):
        metadata = json_dict(row.get("metadata_json"))
        rows.append(
            {
                "Date": row.get("entry_date"),
                "Symbol": json_list_text(row.get("symbols_json")),
                "Strategy": json_list_text(row.get("strategies_json")),
                "Status": metadata.get("status", ""),
                "Direction": metadata.get("direction", ""),
                "Horizon": metadata.get("horizon", ""),
                "Confidence": metadata.get("confidence", ""),
                "Title": row.get("title"),
                "Thesis": row.get("body") or "",
                "Invalidation": metadata.get("invalidation", ""),
                "Expected Outcome": metadata.get("expected_outcome", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def report_files_frame() -> pd.DataFrame:
    rows = []
    for root in REPORT_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*"), key=lambda item: item.stat().st_mtime if item.is_file() else 0, reverse=True):
            if not path.is_file():
                continue
            if path.name == ".DS_Store":
                continue
            rows.append(
                {
                    "File": path.name,
                    "Path": display_path(path),
                    "Kind": path.suffix.lower().lstrip(".") or "file",
                    "Size KB": round(path.stat().st_size / 1024, 1),
                    "Modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                }
            )
            if len(rows) >= 250:
                break
        if len(rows) >= 250:
            break
    return pd.DataFrame(rows, columns=["File", "Path", "Kind", "Size KB", "Modified"])


def save_entry(
    *,
    category: str,
    entry_date: date,
    title: str,
    body: str | None = None,
    environment: str | None = None,
    symbols: str = "",
    strategies: str = "",
    tags: str = "",
    mistake: str | None = None,
    lesson: str | None = None,
    follow_up: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    result = write_journal_entry(
        journal_db,
        entry_date=entry_date,
        category=category,
        title=title,
        body=body,
        environment=None if environment in (None, "all", "") else environment,
        symbols=csv_list(symbols, uppercase=True),
        strategies=csv_list(strategies),
        tags=csv_list(tags),
        mistake=mistake,
        lesson=lesson,
        follow_up=follow_up,
        metadata=metadata,
    )
    st.success(f"Saved {category.replace('_', ' ')} entry: {result.entry_id}")


journal_db = ensure_journal_schema(default_journal_ledger_path())
account_ledger = default_account_ledger_path()
paper_ledger = default_paper_trading_ledger_path()

today = datetime.now(timezone.utc).date()
selected_date = st.sidebar.date_input("Journal date", value=today)
if not isinstance(selected_date, date):
    selected_date = today

entries = load_journal_entries(journal_db, limit=500)
selected_entries = load_journal_entries(journal_db, entry_date=selected_date, limit=500)
live_nav = load_account_nav_history(account_ledger, environment="live")
paper_nav = load_account_nav_history(account_ledger, environment="paper")
live_events_all = load_account_trade_events(account_ledger, environment="live", limit=500)
paper_events_all = load_account_trade_events(account_ledger, environment="paper", limit=500)
paper_orders_all = load_latest_paper_orders(paper_ledger, limit=500)
paper_reviews_all = load_latest_paper_execution_reviews(paper_ledger, limit=500)

live_events = filter_by_date(live_events_all, "occurred_at", selected_date)
paper_events = filter_by_date(paper_events_all, "occurred_at", selected_date)
paper_orders = filter_by_date(paper_orders_all, "created_at", selected_date)
paper_reviews = filter_by_date(paper_reviews_all, "reviewed_at", selected_date)
nav_rows = nav_metric_rows(live_nav, paper_nav, selected_date)
digest = digest_text(
    selected=selected_date,
    nav_rows=nav_rows,
    live_events=live_events,
    paper_events=paper_events,
    paper_orders=paper_orders,
    paper_reviews=paper_reviews,
    journal_entries=selected_entries,
)


page_header(
    title="Journal & Reports",
    title_zh="交易日志与报告",
    subtitle="Daily notes, trade theses, mistake reviews, P&L digest, report references, and later Discord summary archives.",
    subtitle_zh="记录每日笔记、交易假设、错误复盘、P&L 摘要、报告引用与后续 Discord 汇总归档。",
    language=OPS_LANG,
)

top = st.columns(6)
top[0].metric(T("entries_today"), str(len(selected_entries)))
top[1].metric(T("all_entries"), str(len(entries)))
top[2].metric(T("live_pnl", "Live P&L"), signed_money(nav_rows.loc[nav_rows["Account"].eq("Live"), "Daily P&L"].iloc[0] if not nav_rows.empty else None))
top[3].metric(T("paper_pnl", "Paper P&L"), signed_money(nav_rows.loc[nav_rows["Account"].eq("Paper"), "Daily P&L"].iloc[0] if not nav_rows.empty else None))
top[4].metric(T("paper_tickets"), str(len(paper_orders)))
top[5].metric(T("reviews"), str(len(paper_reviews)))

st.caption(T("journal_ledger", path=display_path(journal_db)))

daily_tab, thesis_tab, digest_tab, mistakes_tab, reports_tab = st.tabs(
    ops_tabs(OPS_LANG, "journal_tabs")
)

with daily_tab:
    st.subheader(T("daily_notes"))
    with st.form("daily_note_form"):
        cols = st.columns([1, 1, 1, 1])
        note_date = cols[0].date_input(T("date"), value=selected_date, key="daily_note_date")
        environment = cols[1].selectbox(T("environment"), ["all", "live", "paper", "discretionary", "research"], key="daily_env")
        symbols = cols[2].text_input(T("symbols"), placeholder="AAPL, SPY", key="daily_symbols")
        tags = cols[3].text_input(T("tags"), placeholder="discipline, macro", key="daily_tags")
        title = st.text_input(T("title"), value=T("daily_note_default"))
        body = st.text_area(T("daily_note_body"), height=160)
        follow_up = st.text_area(T("follow_ups"), height=90)
        submitted = st.form_submit_button(T("save_daily_note"), disabled=not title.strip())
    if submitted:
        save_entry(
            category="daily_note",
            entry_date=note_date,
            title=title,
            body=body,
            environment=environment,
            symbols=symbols,
            tags=tags,
            follow_up=follow_up,
        )
        st.rerun()

    st.subheader(T("entries_for_selected_date"))
    display = journal_display(selected_entries)
    if display.empty:
        st.info(T("no_journal_entries"))
    else:
        render_dark_table(display, max_height_px=460)

with thesis_tab:
    st.subheader(T("trade_thesis_log"))
    with st.form("trade_thesis_form"):
        cols = st.columns([1, 1, 1, 1])
        thesis_date = cols[0].date_input(T("date"), value=selected_date, key="thesis_date")
        symbol = cols[1].text_input(T("symbol"), placeholder="AAPL", key="thesis_symbol").upper().strip()
        environment = cols[2].selectbox(T("environment"), ["discretionary", "paper", "live"], key="thesis_env")
        strategy = cols[3].text_input(T("strategy_setup_id"), placeholder="manual_options_watch")
        cols2 = st.columns([1, 1, 1, 1])
        direction = cols2[0].selectbox(T("direction"), ["long", "short", "income", "hedge", "spread", "watch", "avoid"])
        horizon = cols2[1].selectbox(T("horizon"), ["intraday", "days", "weeks", "months", "long-term"])
        status = cols2[2].selectbox(T("status"), ["idea", "watching", "entered", "closed", "rejected", "paper candidate"])
        confidence = cols2[3].slider(T("confidence"), min_value=0, max_value=100, value=50, step=5)
        title = st.text_input(T("thesis_title"), value=f"{symbol} thesis" if symbol else "")
        thesis_body = st.text_area(T("thesis"), height=130)
        invalidation = st.text_area(T("invalidation"), height=90)
        expected_outcome = st.text_area(T("expected_outcome"), height=90)
        risk = st.text_area(T("risk_sizing_notes"), height=90)
        tags = st.text_input(T("tags"), placeholder="earnings, valuation, vol")
        submitted = st.form_submit_button(T("save_thesis"), disabled=not symbol or not title.strip())
    if submitted:
        save_entry(
            category="trade_thesis",
            entry_date=thesis_date,
            title=title,
            body=thesis_body,
            environment=environment,
            symbols=symbol,
            strategies=strategy,
            tags=tags,
            follow_up=risk,
            metadata={
                "direction": direction,
                "horizon": horizon,
                "status": status,
                "confidence": confidence / 100.0,
                "invalidation": invalidation,
                "expected_outcome": expected_outcome,
                "risk": risk,
            },
        )
        st.rerun()

    thesis = thesis_display(entries)
    if thesis.empty:
        st.info(T("no_trade_thesis"))
    else:
        render_dark_table(thesis, max_height_px=460)

with digest_tab:
    st.subheader(T("end_of_day_summary"))
    digest_draft = st.text_area(
        T("generated_digest"),
        value=digest,
        height=220,
        key=f"eod_digest_{selected_date.isoformat()}",
    )
    left, right = st.columns([1, 1])
    with left:
        st.subheader(T("pnl_digest"))
        render_dark_table(nav_rows, max_height_px=360)
    with right:
        st.subheader(T("daily_activity_counts"))
        render_dark_table(
            pd.DataFrame(
                [
                    {"Area": "Live events", "Rows": len(live_events)},
                    {"Area": "Paper events", "Rows": len(paper_events)},
                    {"Area": "Paper tickets", "Rows": len(paper_orders)},
                    {"Area": "Paper reviews", "Rows": len(paper_reviews)},
                    {"Area": "Journal entries", "Rows": len(selected_entries)},
                ]
            ),
            max_height_px=360,
        )
    with st.form("eod_summary_form"):
        title = st.text_input(T("summary_title"), value=f"EOD summary {selected_date.isoformat()}")
        reflection = st.text_area(T("human_reflection"), height=140)
        follow_up = st.text_area(T("tomorrow_follow_ups"), height=90)
        submitted = st.form_submit_button(T("save_eod_summary"))
    if submitted:
        save_entry(
            category="eod_summary",
            entry_date=selected_date,
            title=title,
            body=f"{digest_draft}\n\nReflection:\n{reflection}",
            environment="all",
            tags="eod,digest",
            follow_up=follow_up,
            metadata={
                "live_events": len(live_events),
                "paper_events": len(paper_events),
                "paper_orders": len(paper_orders),
                "paper_reviews": len(paper_reviews),
            },
        )
        st.rerun()

with mistakes_tab:
    st.subheader(T("mistake_review"))
    with st.form("mistake_form"):
        cols = st.columns([1, 1, 1, 1])
        mistake_date = cols[0].date_input(T("date"), value=selected_date, key="mistake_date")
        severity = cols[1].selectbox(T("severity"), ["low", "medium", "high", "process break"])
        environment = cols[2].selectbox(T("environment"), ["discretionary", "paper", "live", "research", "ops"], key="mistake_env")
        symbols = cols[3].text_input(T("symbols"), placeholder="optional", key="mistake_symbols")
        title = st.text_input(T("mistake_title"))
        mistake = st.text_area(T("what_went_wrong"), height=120)
        lesson = st.text_area(T("what_did_i_learn"), height=120)
        follow_up = st.text_area(T("rule_process_change"), height=100)
        tags = st.text_input(T("tags"), placeholder="fomo, sizing, execution")
        submitted = st.form_submit_button(T("save_mistake_review"), disabled=not title.strip())
    if submitted:
        save_entry(
            category="mistake",
            entry_date=mistake_date,
            title=title,
            body=mistake,
            environment=environment,
            symbols=symbols,
            tags=tags,
            mistake=mistake,
            lesson=lesson,
            follow_up=follow_up,
            metadata={"severity": severity},
        )
        st.rerun()

    mistakes = entries[entries["category"].eq("mistake")] if not entries.empty else pd.DataFrame()
    display = journal_display(mistakes)
    if display.empty:
        st.info(T("no_mistake_reviews"))
    else:
        render_dark_table(display, max_height_px=460)

with reports_tab:
    st.subheader(T("exported_reports"))
    files = report_files_frame()
    if files.empty:
        st.info(T("no_report_files"))
    else:
        render_dark_table(files, max_height_px=460)

    st.subheader(T("manual_report_reference"))
    with st.form("report_reference_form"):
        report_date = st.date_input(T("date"), value=selected_date, key="report_date")
        title = st.text_input(T("report_title"))
        path = st.text_input(T("path_or_url"))
        summary = st.text_area(T("why_report_matters"), height=100)
        tags = st.text_input(T("tags"), placeholder="discord, paper, live, weekly")
        submitted = st.form_submit_button(T("save_report_reference"), disabled=not title.strip())
    if submitted:
        save_entry(
            category="report_reference",
            entry_date=report_date,
            title=title,
            body=summary,
            environment="all",
            tags=tags,
            metadata={"path_or_url": path},
        )
        st.rerun()

    st.subheader(T("saved_report_references"))
    refs = entries[entries["category"].eq("report_reference")] if not entries.empty else pd.DataFrame()
    display = journal_display(refs)
    if display.empty:
        st.info(T("no_report_references"))
    else:
        render_dark_table(display, max_height_px=460)
