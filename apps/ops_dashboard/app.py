"""Operations command center for Alpha Factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import default_account_ledger_path  # noqa: E402
from oqp.config import load_settings  # noqa: E402
from oqp.ops import collect_ops_status  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def status_label(status: str) -> str:
    return status.upper()


def money(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def status_dataframe(rows: list[dict[str, object]]) -> None:
    if not rows:
        st.info("No status rows available.")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Detail": st.column_config.TextColumn("Detail", width="large"),
        },
    )


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


def account_signal(frame: pd.DataFrame, environment: str) -> str:
    if frame.empty or "environment" not in frame:
        return "No account snapshot."
    rows = frame[frame["environment"].eq(environment)]
    if rows.empty:
        return "No account snapshot."
    row = rows.iloc[0]
    return (
        f"NAV {money(row.get('net_liquidation'))}; "
        f"cash {money(row.get('cash'))}; "
        f"positions {int(row.get('position_count') or 0)}; "
        f"age {float(row.get('age_hours') or 0.0):.1f}h"
    )


def status_detail(rows: pd.DataFrame, fallback: str) -> str:
    if rows.empty:
        return fallback
    failed = rows[rows["Status"].eq("fail")]
    warned = rows[rows["Status"].eq("warn")]
    source = failed if not failed.empty else warned
    if source.empty:
        return fallback
    first = source.iloc[0]
    return f"{first.get('Check')}: {first.get('Detail')}"


def pipeline_rows(items: pd.DataFrame, accounts: pd.DataFrame) -> list[dict[str, str]]:
    paper = check_subset(items, contains=("Paper",))
    live = check_subset(items, contains=("Live", "Portfolio Snapshot"))
    safety = check_subset(items, categories=("Safety",))
    ops = check_subset(items, categories=("Schedulers", "Host", "Notifications"))

    return [
        {
            "Layer": "Research",
            "Mode": "Manual",
            "Status": "LOCAL",
            "Signal": "Trial ledger and candidate export stay in the research dashboard.",
        },
        {
            "Layer": "Paper Trading",
            "Mode": "Automated",
            "Status": status_label(worst_status(paper)),
            "Signal": status_detail(paper, account_signal(accounts, "paper")),
        },
        {
            "Layer": "Live Monitoring",
            "Mode": "Read-only",
            "Status": status_label(worst_status(live)),
            "Signal": status_detail(live, account_signal(accounts, "live")),
        },
        {
            "Layer": "Execution Safety",
            "Mode": "Locked",
            "Status": status_label(worst_status(safety)),
            "Signal": status_detail(safety, "Live trading disabled; paper submission gated."),
        },
        {
            "Layer": "Ops And Alerts",
            "Mode": "Monitoring",
            "Status": status_label(worst_status(ops)),
            "Signal": status_detail(ops, "Schedulers, host, and Discord checks are stable."),
        },
    ]


st.set_page_config(page_title="Ops Dashboard", layout="wide", page_icon="OPS")

settings = load_settings()
snapshot = collect_ops_status(settings=settings)
items_df = pd.DataFrame(snapshot.item_rows)
account_df = pd.DataFrame(snapshot.account_rows)
event_df = pd.DataFrame(snapshot.event_rows)
account_ledger_path = default_account_ledger_path()

st.title("Ops Dashboard")
st.caption(f"Checked at {snapshot.checked_at.isoformat(timespec='seconds')}")

overall_counts = items_df["Status"].value_counts().to_dict() if not items_df.empty else {}
top_cols = st.columns(6)
top_cols[0].metric("Overall", status_label(snapshot.overall_status))
top_cols[1].metric("Failures", str(overall_counts.get("fail", 0)))
top_cols[2].metric("Warnings", str(overall_counts.get("warn", 0)))
top_cols[3].metric("Checks", str(len(items_df)))
top_cols[4].metric("Accounts", str(len(account_df)))
top_cols[5].metric("Events", str(len(event_df)))

st.divider()

if not items_df.empty:
    failed_or_warn = items_df[items_df["Status"].isin(["fail", "warn"])]
else:
    failed_or_warn = pd.DataFrame()

overview_tab, accounts_tab, plumbing_tab, jobs_tab, raw_tab = st.tabs(
    ["Overview", "Accounts", "Gateways", "Jobs", "Raw Checks"]
)

with overview_tab:
    st.subheader("Operating Pipeline")
    st.dataframe(
        pd.DataFrame(pipeline_rows(items_df, account_df)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Mode": st.column_config.TextColumn("Mode", width="small"),
            "Signal": st.column_config.TextColumn("Signal", width="large"),
        },
    )

    st.subheader("Attention")
    if failed_or_warn.empty:
        st.success("All collected checks are passing.")
    else:
        status_dataframe(failed_or_warn.to_dict("records"))

with accounts_tab:
    st.subheader("Account Snapshots")
    if account_df.empty:
        st.warning(f"No unified account snapshots found in {display_path(account_ledger_path)}.")
    else:
        account_view = account_df.copy()
        account_view["NAV"] = account_view["net_liquidation"].map(money)
        account_view["Cash"] = account_view["cash"].map(money)
        account_view["Daily P&L"] = account_view["daily_pnl"].map(money)
        account_view["Age Hours"] = account_view["age_hours"].map(
            lambda value: "" if value is None else f"{float(value):.2f}"
        )
        account_view = account_view[
            [
                "environment",
                "profile",
                "account_id",
                "as_of",
                "NAV",
                "Cash",
                "Daily P&L",
                "position_count",
                "Age Hours",
            ]
        ].rename(
            columns={
                "environment": "Environment",
                "profile": "Profile",
                "account_id": "Account",
                "as_of": "As Of",
                "position_count": "Positions",
            }
        )
        st.dataframe(account_view, use_container_width=True, hide_index=True)
    st.caption(f"Account ledger: {display_path(account_ledger_path)}")

    st.subheader("Recent Account Events")
    if event_df.empty:
        st.info("No account trade events have been recorded yet.")
    else:
        event_view = event_df.copy()
        event_view["Quantity"] = event_view["quantity"].map(
            lambda value: "" if value is None else f"{float(value):,.4f}"
        )
        event_view["Price"] = event_view["price"].map(money)
        event_view = event_view[
            [
                "occurred_at",
                "environment",
                "event_type",
                "symbol",
                "side",
                "Quantity",
                "Price",
                "strategy_id",
                "order_id",
            ]
        ].rename(
            columns={
                "occurred_at": "Occurred",
                "environment": "Environment",
                "event_type": "Event",
                "symbol": "Symbol",
                "side": "Side",
                "strategy_id": "Strategy",
                "order_id": "Order / Proposal",
            }
        )
        st.dataframe(event_view, use_container_width=True, hide_index=True)

with plumbing_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Gateway Sockets")
        status_dataframe(category_rows(items_df, "Gateway"))

    with right:
        st.subheader("IBKR API Heartbeats")
        status_dataframe(category_rows(items_df, "Broker Heartbeat"))

    st.subheader("Safety Gates")
    status_dataframe(category_rows(items_df, "Safety"))

with jobs_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Schedulers And Jobs")
        status_dataframe(category_rows(items_df, "Schedulers") + category_rows(items_df, "Jobs"))

    with right:
        st.subheader("Notifications")
        status_dataframe(category_rows(items_df, "Notifications"))

    st.subheader("Host Health")
    host_cols = st.columns(4)
    host_cols[0].metric(
        "Disk Used",
        f"{float(snapshot.host_summary.get('disk_used_pct', 0.0)) * 100:.1f}%",
    )
    host_cols[1].metric(
        "Disk Free",
        f"{float(snapshot.host_summary.get('disk_free_gb', 0.0)):.1f} GB",
    )
    memory_used = snapshot.host_summary.get("memory_used_pct")
    host_cols[2].metric(
        "Memory Used",
        "n/a" if memory_used is None else f"{float(memory_used) * 100:.1f}%",
    )
    host_cols[3].metric(
        "Memory Free",
        "n/a"
        if snapshot.host_summary.get("memory_free_gb") is None
        else f"{float(snapshot.host_summary.get('memory_free_gb')):.1f} GB",
    )
    status_dataframe(category_rows(items_df, "Host"))

with raw_tab:
    status_dataframe(snapshot.item_rows)
