"""QMT connector UI/data helpers for the Ops dashboard."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from oqp.config.paths import REPO_ROOT


DEFAULT_WINDOWS_QMT_AUDIT_PATH = REPO_ROOT / "runtime" / "logs" / "windows_qmt_connector_audit.jsonl"


def qmt_submit_state(settings: Any) -> str:
    if getattr(settings, "allow_qmt_live_trading", False):
        return "live armed"
    if getattr(settings, "allow_qmt_paper_order_submit", False):
        return "paper armed"
    return "locked"


def qmt_overall_status(snapshot: Any) -> str:
    frame = qmt_status_frame(snapshot)
    if frame.empty:
        return "missing"
    order = {"fail": 3, "warn": 2, "pass": 1}
    worst = max(
        (str(status).lower() for status in frame["Status"].tolist()),
        key=lambda status: order.get(status, 0),
    )
    return worst or "missing"


def qmt_status_frame(snapshot: Any) -> pd.DataFrame:
    columns = ["Category", "Check", "Status", "Detail"]
    frame = pd.DataFrame(getattr(snapshot, "item_rows", []) or [])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    text = (
        frame["Category"].astype(str)
        + " "
        + frame["Check"].astype(str)
        + " "
        + frame["Detail"].astype(str)
    )
    mask = text.str.contains("QMT", case=False, na=False)
    return frame.loc[mask].reindex(columns=columns).reset_index(drop=True)


def qmt_account_rows(snapshot: Any, *, environment: str | None = None) -> pd.DataFrame:
    columns = [
        "Environment",
        "Broker",
        "Profile",
        "Account",
        "NAV",
        "Cash",
        "Daily P&L",
        "Positions",
        "As Of",
        "Age Hours",
    ]
    frame = pd.DataFrame(getattr(snapshot, "account_rows", []) or [])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _qmt_filter(frame)
    if environment:
        frame = frame.loc[frame.get("environment", "").astype(str).str.lower().eq(environment.lower())]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "Environment": row.get("environment"),
                "Broker": row.get("broker"),
                "Profile": row.get("profile"),
                "Account": row.get("account_id"),
                "NAV": _money(row.get("net_liquidation")),
                "Cash": _money(row.get("cash")),
                "Daily P&L": _signed_money(row.get("daily_pnl")),
                "Positions": row.get("position_count"),
                "As Of": row.get("as_of"),
                "Age Hours": _number_text(row.get("age_hours")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def qmt_position_slice(positions: pd.DataFrame, *, environment: str | None = None) -> pd.DataFrame:
    columns = [
        "Environment",
        "Profile",
        "Symbol",
        "Asset Class",
        "Quantity",
        "Market Value",
        "Unrealized P&L",
        "Currency",
        "As Of",
    ]
    frame = pd.DataFrame() if positions is None else pd.DataFrame(positions).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _qmt_filter(frame)
    if environment and "environment" in frame:
        frame = frame.loc[frame["environment"].astype(str).str.lower().eq(environment.lower())]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            {
                "Environment": row.get("environment"),
                "Profile": row.get("profile"),
                "Symbol": row.get("symbol"),
                "Asset Class": row.get("asset_class"),
                "Quantity": _number_text(row.get("quantity")),
                "Market Value": _money(row.get("market_value")),
                "Unrealized P&L": _signed_money(row.get("unrealized_pnl")),
                "Currency": row.get("currency"),
                "As Of": row.get("as_of"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def qmt_exposure_by_asset(positions: pd.DataFrame) -> pd.DataFrame:
    columns = ["Environment", "Asset Class", "Market Value", "Unrealized P&L", "Rows"]
    frame = pd.DataFrame() if positions is None else pd.DataFrame(positions).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = _qmt_filter(frame)
    if frame.empty or "asset_class" not in frame:
        return pd.DataFrame(columns=columns)
    for column in ("market_value", "unrealized_pnl"):
        frame[column] = pd.to_numeric(frame.get(column, 0.0), errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(["environment", "asset_class"], dropna=False)
        .agg(
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
            rows=("symbol", "count"),
        )
        .reset_index()
    )
    grouped["Market Value"] = grouped["market_value"].map(_money)
    grouped["Unrealized P&L"] = grouped["unrealized_pnl"].map(_signed_money)
    grouped = grouped.rename(
        columns={
            "environment": "Environment",
            "asset_class": "Asset Class",
            "rows": "Rows",
        }
    )
    return grouped.reindex(columns=columns)


def qmt_safety_gate_frame(settings: Any) -> pd.DataFrame:
    read_url = str(getattr(settings, "qmt_connector_url", "") or "").rstrip("/")
    submit_url = str(getattr(settings, "qmt_submit_connector_url", "") or "").rstrip("/")
    qmt_submit_armed = bool(
        getattr(settings, "allow_qmt_paper_order_submit", False)
        or getattr(settings, "allow_qmt_live_trading", False)
    )
    rows = [
        _gate(
            "Connector enabled",
            "enabled" if getattr(settings, "qmt_connector_enabled", False) else "locked",
            "pass",
            f"QMT_CONNECTOR_ENABLED={str(getattr(settings, 'qmt_connector_enabled', False)).lower()}",
        ),
        _gate(
            "Private connector required",
            "required" if getattr(settings, "qmt_require_private_connector", True) else "disabled",
            "pass" if getattr(settings, "qmt_require_private_connector", True) else "fail",
            f"QMT_REQUIRE_PRIVATE_CONNECTOR={str(getattr(settings, 'qmt_require_private_connector', True)).lower()}",
        ),
        _gate(
            "Read-only connector URL",
            "configured" if read_url else "missing",
            "pass" if read_url else "fail",
            read_url or "missing",
        ),
        _gate(
            "Submit connector isolated",
            "isolated" if read_url and submit_url and read_url != submit_url else "shared",
            "pass" if not qmt_submit_armed or (read_url and submit_url and read_url != submit_url) else "fail",
            submit_url or "missing",
        ),
        _gate(
            "API token",
            "configured" if getattr(settings, "qmt_api_token", None) else "missing",
            "pass" if (not qmt_submit_armed or getattr(settings, "qmt_api_token", None)) else "fail",
            "Required when QMT submit is armed.",
        ),
        _gate(
            "HMAC signing",
            "configured" if getattr(settings, "qmt_request_signing_secret", None) else "missing",
            "pass" if (not qmt_submit_armed or getattr(settings, "qmt_request_signing_secret", None)) else "fail",
            "Required when QMT submit is armed.",
        ),
        _gate(
            "Paper submit",
            "armed" if getattr(settings, "allow_qmt_paper_order_submit", False) else "locked",
            "warn" if getattr(settings, "allow_qmt_paper_order_submit", False) else "pass",
            f"ALLOW_QMT_PAPER_ORDER_SUBMIT={str(getattr(settings, 'allow_qmt_paper_order_submit', False)).lower()}",
        ),
        _gate(
            "Live submit",
            "armed" if getattr(settings, "allow_qmt_live_trading", False) else "locked",
            "fail" if getattr(settings, "allow_qmt_live_trading", False) else "pass",
            f"ALLOW_QMT_LIVE_TRADING={str(getattr(settings, 'allow_qmt_live_trading', False)).lower()}",
        ),
    ]
    return pd.DataFrame(rows, columns=["Gate", "State", "Status", "Detail"])


def qmt_connector_contract_frame(settings: Any) -> pd.DataFrame:
    rows = [
        {"Contract": "Read connector", "Value": getattr(settings, "qmt_connector_url", "")},
        {"Contract": "Submit connector", "Value": getattr(settings, "qmt_submit_connector_url", "")},
        {"Contract": "Account type", "Value": getattr(settings, "qmt_account_type", "")},
        {"Contract": "Session ID", "Value": getattr(settings, "qmt_session_id", "")},
        {"Contract": "Timeout seconds", "Value": getattr(settings, "qmt_timeout_seconds", "")},
        {"Contract": "OQP audit log", "Value": _display_path(getattr(settings, "qmt_audit_log_path", ""))},
        {"Contract": "Windows audit log", "Value": _display_path(DEFAULT_WINDOWS_QMT_AUDIT_PATH)},
    ]
    return pd.DataFrame(rows, columns=["Contract", "Value"])


def qmt_strategy_route_frame(registry: pd.DataFrame) -> pd.DataFrame:
    columns = ["Strategy", "Market", "QMT Route", "Status", "Allowed Symbols", "Notes"]
    frame = pd.DataFrame() if registry is None else pd.DataFrame(registry).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        symbols = _json_list(row.get("allowed_symbols_json"))
        market = str(row.get("market_vertical") or "")
        qmt_candidate = _market_or_symbols_look_qmt(market, symbols)
        rows.append(
            {
                "Strategy": row.get("strategy_id"),
                "Market": market,
                "QMT Route": "candidate" if qmt_candidate else "not primary",
                "Status": row.get("status"),
                "Allowed Symbols": ", ".join(symbols),
                "Notes": row.get("notes"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def qmt_route_candidate_frame(symbol: str, settings: Any | None = None) -> pd.DataFrame:
    columns = ["Field", "Value"]
    text = str(symbol or "").upper().strip()
    qmt_like = _symbol_looks_qmt(text)
    rows = [
        {"Field": "Symbol", "Value": text or "missing"},
        {"Field": "QMT Route Candidate", "Value": "yes" if qmt_like else "not by symbol"},
        {"Field": "Likely Lane", "Value": _qmt_lane_for_symbol(text)},
    ]
    if settings is not None:
        rows.extend(
            [
                {"Field": "Read Profile", "Value": "qmt_paper_readonly"},
                {"Field": "Submit Profile", "Value": "qmt_paper_submit"},
                {"Field": "Submit State", "Value": qmt_submit_state(settings)},
            ]
        )
    return pd.DataFrame(rows, columns=columns)


def qmt_audit_path_frame(settings: Any, *, extra_paths: Iterable[str | Path] = ()) -> pd.DataFrame:
    rows = []
    for label, path in qmt_audit_paths(settings, extra_paths=extra_paths):
        rows.append(
            {
                "Log": label,
                "Path": _display_path(path),
                "Status": "present" if Path(path).exists() else "missing",
            }
        )
    return pd.DataFrame(rows, columns=["Log", "Path", "Status"])


def qmt_audit_paths(
    settings: Any,
    *,
    extra_paths: Iterable[str | Path] = (),
) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    configured = getattr(settings, "qmt_audit_log_path", None)
    if configured:
        paths.append(("OQP audit", Path(configured)))
    paths.append(("Windows connector audit", DEFAULT_WINDOWS_QMT_AUDIT_PATH))
    for path in extra_paths:
        paths.append(("Extra audit", Path(path)))
    unique: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        unique.append((label, resolved))
        seen.add(resolved)
    return unique


def qmt_audit_events_frame(
    settings: Any,
    *,
    extra_paths: Iterable[str | Path] = (),
    limit: int = 20,
) -> pd.DataFrame:
    columns = ["Time", "Log", "Event", "Endpoint", "Status", "Account", "Symbol", "Message"]
    events: list[dict[str, Any]] = []
    for label, path in qmt_audit_paths(settings, extra_paths=extra_paths):
        events.extend(_read_audit_events(path, label))
    if not events:
        return pd.DataFrame(columns=columns)
    events.sort(key=lambda row: str(row.get("Time") or ""), reverse=True)
    return pd.DataFrame(events[:limit], columns=columns)


def qmt_audit_count(settings: Any) -> int:
    return int(len(qmt_audit_events_frame(settings, limit=10_000)))


def render_qmt_connector_panel(settings: Any, snapshot: Any | None = None, *, compact: bool = False) -> None:
    import streamlit as st

    from oqp.ui.streamlit_theme import render_dark_table

    cols = st.columns(4)
    cols[0].metric("QMT Connector", "enabled" if getattr(settings, "qmt_connector_enabled", False) else "locked")
    cols[1].metric("QMT Heartbeat", qmt_overall_status(snapshot) if snapshot is not None else "n/a")
    cols[2].metric("QMT Submit", qmt_submit_state(settings))
    cols[3].metric("QMT Account Type", str(getattr(settings, "qmt_account_type", "STOCK")))
    if snapshot is not None:
        render_dark_table(
            qmt_status_frame(snapshot),
            empty_message="No QMT status rows are available.",
            max_height_px=220 if compact else 360,
        )


def render_qmt_safety_panel(settings: Any, *, max_height_px: int = 360) -> None:
    from oqp.ui.streamlit_theme import render_dark_table

    render_dark_table(qmt_safety_gate_frame(settings), max_height_px=max_height_px)


def render_qmt_account_panel(
    snapshot: Any,
    positions: pd.DataFrame | None = None,
    *,
    environment: str | None = None,
) -> None:
    from oqp.ui.streamlit_theme import render_dark_table

    render_dark_table(
        qmt_account_rows(snapshot, environment=environment),
        empty_message="No QMT account rows are available yet.",
        max_height_px=260,
    )
    if positions is not None:
        render_dark_table(
            qmt_position_slice(positions, environment=environment),
            empty_message="No QMT position rows are available yet.",
            max_height_px=360,
        )


def render_qmt_audit_panel(settings: Any, *, limit: int = 20) -> None:
    from oqp.ui.streamlit_theme import render_dark_table

    render_dark_table(qmt_audit_path_frame(settings), max_height_px=180)
    render_dark_table(
        qmt_audit_events_frame(settings, limit=limit),
        empty_message="No QMT audit events are available yet.",
        max_height_px=360,
    )


def _gate(gate: str, state: str, status: str, detail: str) -> dict[str, str]:
    return {"Gate": gate, "State": state, "Status": status, "Detail": detail}


def _qmt_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    text_parts = []
    for column in ("broker", "profile", "account_key", "metadata_json"):
        if column in frame:
            text_parts.append(frame[column].astype(str))
    if not text_parts:
        return frame.iloc[0:0].copy()
    text = text_parts[0]
    for part in text_parts[1:]:
        text = text + " " + part
    return frame.loc[text.str.contains("qmt", case=False, na=False)].copy()


def _read_audit_events(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        request = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        symbol = request.get("symbol") or response.get("symbol")
        if not symbol and isinstance(response.get("order"), dict):
            symbol = response["order"].get("symbol")
        message = payload.get("error") or response.get("message") or response.get("error")
        rows.append(
            {
                "Time": payload.get("ts") or payload.get("created_at"),
                "Log": label,
                "Event": payload.get("event"),
                "Endpoint": payload.get("endpoint"),
                "Status": payload.get("status_code") or payload.get("status"),
                "Account": payload.get("account_id") or request.get("account_id"),
                "Symbol": symbol,
                "Message": message,
            }
        )
    return rows


def _json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _market_or_symbols_look_qmt(market: str, symbols: list[str]) -> bool:
    text = market.upper()
    if any(token in text for token in ("CN", "CHINA", "FUTURES_CN", "EQUITY_CN", "OPTIONS_CN")):
        return True
    return any(_symbol_looks_qmt(symbol) for symbol in symbols)


def _symbol_looks_qmt(symbol: str) -> bool:
    text = str(symbol or "").upper().strip()
    return text.endswith((".SH", ".SZ", ".BJ", ".CFE", ".INE", ".SHF", ".DCE", ".CZC", ".GFEX", ".SF"))


def _qmt_lane_for_symbol(symbol: str) -> str:
    text = str(symbol or "").upper().strip()
    if text.endswith((".SH", ".SZ", ".BJ")):
        return "EQUITY_CN"
    if text.endswith((".CFE", ".INE", ".SHF", ".DCE", ".CZC", ".GFEX", ".SF")):
        return "FUTURES_CN"
    return "unclassified"


def _display_path(value: Any) -> str:
    if value in (None, ""):
        return "missing"
    path = Path(value)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _money(value: Any) -> str:
    number = _float(value)
    return "missing" if number is None else f"{number:,.2f}"


def _signed_money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    sign = "+" if number > 0 else ""
    return f"{sign}{number:,.2f}"


def _number_text(value: Any) -> str:
    number = _float(value)
    return "missing" if number is None else f"{number:g}"


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed
