"""Live Portfolio page for the unified Ops dashboard."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode
except ImportError:  # pragma: no cover - optional dashboard dependency
    AgGrid = None
    DataReturnMode = None
    GridOptionsBuilder = None
    GridUpdateMode = None
    JsCode = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oqp.accounts import (  # noqa: E402
    DEFAULT_MANUAL_EXTERNAL_INPUT_PATH,
    DEFAULT_LIVE_BROKER_PROFILE,
    UNIFIED_LIVE_PROFILE,
    account_nav_drawdowns,
    account_performance_summary,
    account_trade_events_display,
    account_trade_event_summary,
    default_account_ledger_path,
    load_account_nav_history,
    load_account_position_history,
    load_account_trade_events,
    load_latest_account_nav,
    load_latest_account_positions,
    load_manual_external_positions_as_account_positions,
    materialize_unified_live_account_snapshot,
    sync_manual_external_positions_from_json,
    write_manual_external_positions_file,
    blended_live_nav_history,
)
from oqp.market import (  # noqa: E402
    commodity_channel_index,
    historical_volatility_frame,
    load_cached_market_history,
    load_cached_price_history,
    load_price_history,
    market_cache_status,
    refresh_fmp_market_cache,
)
from oqp.market.volatility import bollinger_squeeze_metrics  # noqa: E402
from oqp.options import (  # noqa: E402
    extract_portfolio_option_legs,
    option_book_summary,
    option_leg_report,
    option_payoff_curve,
    option_payoff_surface,
    option_position_diagnostics,
    option_risk_summary,
    recognize_option_spreads,
    underlying_exposure_report,
)
from oqp.ops import collect_ops_status  # noqa: E402
from oqp.portfolio import (  # noqa: E402
    DEFAULT_HOLDING_PLAN_PATH,
    HOLDING_STYLES,
    PORTFOLIO_TICKER_ALIASES,
    add_holding_plan_columns,
    asset_sleeve_mix,
    concentration_diagnostics_frame,
    currency_exposure_frame,
    default_portfolio_ledger_path,
    enriched_live_holdings,
    load_historical_nav,
    load_holding_styles,
    position_risk_frame,
    save_holding_styles,
    sector_exposure_frame,
)
from oqp.portfolio.live_reporting import CASH_EQUIVALENT_SYMBOLS  # noqa: E402
from oqp.risk import (  # noqa: E402
    MarketRiskConfig,
    combine_price_histories,
    compute_market_risk_decomposition,
)
from oqp.config import load_settings  # noqa: E402
from oqp.ui import (  # noqa: E402
    apply_ops_theme,
    language_selector,
    ops_tabs,
    ops_text,
    page_header,
    qmt_connector_contract_frame,
    qmt_exposure_by_asset,
    qmt_position_slice,
    render_dark_bar_chart,
    render_dark_line_chart,
    render_dark_pie_chart,
    render_dark_table,
    render_qmt_account_panel,
    render_qmt_audit_panel,
    render_qmt_connector_panel,
    style_dark_plotly,
)


st.set_page_config(
    page_title="Live Portfolio",
    layout="wide",
    page_icon="LIVE",
    initial_sidebar_state="expanded",
)

OPS_LANG = language_selector()
apply_ops_theme()

REPORTING_CURRENCY = "USD"
US_ACCOUNT_SCOPE = "US Accounts (IBKR)"
CN_ACCOUNT_SCOPE = "CN Accounts (华源证券 / QMT)"
ACCOUNT_SCOPES = (US_ACCOUNT_SCOPE, CN_ACCOUNT_SCOPE)


def T(key: str, default: str | None = None, **format_values: Any) -> str:
    return ops_text(OPS_LANG, key, default, **format_values)


def currency_header(label: str, currency: str = REPORTING_CURRENCY) -> str:
    return f"{label} ({currency})"


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


def human_timestamp(value: Any) -> str:
    if _missing(value):
        return "missing"
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %d %H:%M UTC")


def human_date(value: Any) -> str:
    if _missing(value):
        return "missing"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%b %d, %Y")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def quantity(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    if abs(number - round(number)) < 1e-8:
        return f"{number:,.0f}"
    return f"{number:,.2f}"


def decimal(value: Any, digits: int = 2) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    return f"{number:,.{digits}f}"


def ratio(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    return f"{number:.2f}x"


def maybe_percent(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "missing"
    return f"{number * 100:.1f}%"


def format_holdings_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a dashboard-friendly copy of holdings-like tables."""

    if frame.empty:
        return frame

    out = frame.copy()
    formatters = {
        "Quantity": quantity,
        "Market Price": money,
        "Market Value": money,
        "Average Cost": money,
        "Unrealized P&L": signed_money,
        "Realized P&L": signed_money,
        "Weight": maybe_percent,
        "HV 5D": maybe_percent,
        "HV 20D": maybe_percent,
        "CCI 20": lambda value: decimal(value, 1),
        "BB Width": maybe_percent,
        "BB 6M %ile": maybe_percent,
        "BB Z": lambda value: decimal(value, 2),
        "As Of": human_timestamp,
    }
    for column, formatter in formatters.items():
        if column in out.columns:
            out[column] = out[column].map(formatter)
    if "Squeeze" in out.columns:
        out["Squeeze"] = out["Squeeze"].map(lambda value: "yes" if bool(value) else "")
    out = out.rename(
        columns={
            "Market Price": currency_header("Market Price"),
            "Market Value": currency_header("Market Value"),
            "Average Cost": currency_header("Average Cost"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
            "Realized P&L": currency_header("Realized P&L"),
        }
    )
    return out


def holdings_aggrid_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare current holdings for an interactive AgGrid table."""

    if frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    if "As Of" in out:
        out["As Of"] = out["As Of"].map(human_timestamp)

    numeric_columns = [
        "Quantity",
        "Market Price",
        "Market Value",
        "Average Cost",
        "Unrealized P&L",
        "Realized P&L",
        "HV 5D",
        "HV 20D",
        "CCI 20",
        "BB Width",
        "BB 6M %ile",
        "BB Z",
        "TP 1x ATR",
        "SL 1x ATR",
    ]
    for column in numeric_columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "Squeeze" in out:
        out["Squeeze"] = out["Squeeze"].map(lambda value: "yes" if bool(value) else "")

    columns = [
        "Symbol",
        "Quantity",
        "Market Value",
        "Holding Style",
        "Average Cost",
        "Unrealized P&L",
        "Market Price",
        "TP 1x ATR",
        "SL 1x ATR",
        "Broker",
        "Asset Class",
        "Underlying",
        "HV 5D",
        "HV 20D",
        "CCI 20",
        "BB Width",
        "BB 6M %ile",
        "BB Z",
        "Squeeze",
        "As Of",
        "Native Currency",
        "Currency",
        "Plan Key",
    ]
    return out[[column for column in columns if column in out.columns]]


def holdings_aggrid_key(frame: pd.DataFrame) -> str:
    """Return a component key that changes whenever displayed grid data changes.

    streamlit-aggrid updates rowData synchronously on an existing component. With
    a frequently refreshed live holdings frame, that update can land while AG Grid
    is drawing rows and raise AG Grid error #252. A data-aware key remounts the
    read-only grid for a new snapshot instead of mutating the rendering instance.
    """

    row_hashes = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    digest = hashlib.sha256(row_hashes).hexdigest()[:16]
    return f"live_holdings_aggrid_{digest}"


def render_holdings_aggrid(frame: pd.DataFrame) -> bool:
    """Render current holdings in an interactive dark AgGrid."""

    if AgGrid is None or GridOptionsBuilder is None or GridUpdateMode is None or DataReturnMode is None or JsCode is None:
        return False

    grid_frame = holdings_aggrid_frame(frame)
    if grid_frame.empty:
        st.info("No live holdings rows match the current filters.")
        return True

    money_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            return Number(params.value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        """
    )
    atr_price_formatter = JsCode(
        """
        function(params) {
            const symbol = String((params.data && params.data.Symbol) || "").toUpperCase();
            const asset = String((params.data && params.data["Asset Class"]) || "").toLowerCase();
            if (asset.includes("cash") || symbol === "CASH" || symbol.endsWith(" CASH")) return "N/A";
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            return Number(params.value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        """
    )
    signed_money_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            const value = Number(params.value);
            const sign = value > 0 ? "+" : "";
            return sign + value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        """
    )
    quantity_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            const value = Number(params.value);
            if (Math.abs(value - Math.round(value)) < 1e-8) return value.toLocaleString(undefined, {maximumFractionDigits: 0});
            return value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        }
        """
    )
    percent_formatter = JsCode(
        """
        function(params) {
            const symbol = String((params.data && params.data.Symbol) || "").toUpperCase();
            const asset = String((params.data && params.data["Asset Class"]) || "").toLowerCase();
            if (asset.includes("cash") || symbol === "CASH" || symbol.endsWith(" CASH")) return "N/A";
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            return (Number(params.value) * 100).toFixed(1) + "%";
        }
        """
    )
    number_formatter = JsCode(
        """
        function(params) {
            const symbol = String((params.data && params.data.Symbol) || "").toUpperCase();
            const asset = String((params.data && params.data["Asset Class"]) || "").toLowerCase();
            if (asset.includes("cash") || symbol === "CASH" || symbol.endsWith(" CASH")) return "N/A";
            if (params.value === null || params.value === undefined || isNaN(params.value)) return "missing";
            return Number(params.value).toFixed(1);
        }
        """
    )
    diagnostic_text_formatter = JsCode(
        """
        function(params) {
            const symbol = String((params.data && params.data.Symbol) || "").toUpperCase();
            const asset = String((params.data && params.data["Asset Class"]) || "").toLowerCase();
            if (asset.includes("cash") || symbol === "CASH" || symbol.endsWith(" CASH")) return "N/A";
            if (params.value === null || params.value === undefined || params.value === "") return "missing";
            return String(params.value);
        }
        """
    )
    symbol_style = JsCode(
        """
        function(params) {
            return {
                color: "#dbeafe",
                fontWeight: "800"
            };
        }
        """
    )
    pnl_style = JsCode(
        """
        function(params) {
            const value = Number(params.value);
            if (isNaN(value)) return { color: "#94a3b8" };
            return { color: value >= 0 ? "#5eead4" : "#fb7185", fontWeight: "800" };
        }
        """
    )
    builder = GridOptionsBuilder.from_dataframe(grid_frame)
    builder.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
        menuTabs=["filterMenuTab", "generalMenuTab"],
    )
    builder.configure_selection(selection_mode="single", use_checkbox=False)
    builder.configure_grid_options(
        rowHeight=38,
        headerHeight=44,
        suppressCellFocus=False,
        enableCellTextSelection=True,
        animateRows=False,
        tooltipShowDelay=250,
        suppressRowClickSelection=True,
        alwaysShowHorizontalScroll=True,
        # The Streamlit component's nested scrolling viewport can miscalculate
        # the virtual row window and leave most holdings as a blank grid area.
        # Holdings is a small table, so rendering every row is both safe and
        # more reliable here. The snapshot-aware component key above prevents
        # live-data updates from mutating a grid that is still rendering.
        suppressRowVirtualisation=True,
        suppressColumnVirtualisation=True,
        ensureDomOrder=True,
        singleClickEdit=True,
        stopEditingWhenCellsLoseFocus=True,
    )
    builder.configure_column("Symbol", pinned="left", headerName="Symbol", width=124, minWidth=112, cellStyle=symbol_style, tooltipField="Symbol")
    if "Holding Style" in grid_frame:
        builder.configure_column(
            "Holding Style",
            headerName="Style",
            headerTooltip="Click a Style cell and choose Trading or Investing; the selection is saved immediately.",
            editable=True,
            cellEditor="agSelectCellEditor",
            cellEditorPopup=True,
            cellEditorParams={"values": ["Investing", "Trading"]},
            width=126,
            minWidth=118,
        )
    if "Plan Key" in grid_frame:
        builder.configure_column("Plan Key", hide=True)
    if "Native Currency" in grid_frame:
        builder.configure_column("Native Currency", headerName="Native CCY", width=116, minWidth=108, tooltipField="Native Currency")
    if "Currency" in grid_frame:
        builder.configure_column("Currency", headerName="Report CCY", width=116, minWidth=108, tooltipField="Currency")
    if "Broker" in grid_frame:
        builder.configure_column("Broker", headerName="Broker", width=124, minWidth=112)
    builder.configure_column("Asset Class", headerName="Asset", width=118, minWidth=108)
    builder.configure_column("Quantity", headerName="Qty", type=["numericColumn"], valueFormatter=quantity_formatter, width=92, minWidth=84)
    for column in ["Market Price", "Market Value", "Average Cost"]:
        if column in grid_frame:
            label = {
                "Market Price": "Mkt Price",
                "Market Value": "Mkt Value",
                "Average Cost": "Avg Cost",
            }.get(column, column)
            builder.configure_column(
                column,
                headerName=label,
                headerTooltip=currency_header(label),
                type=["numericColumn"],
                valueFormatter=money_formatter,
                width=130,
                minWidth=124,
            )
    for column in ["TP 1x ATR", "SL 1x ATR"]:
        if column in grid_frame:
            builder.configure_column(
                column,
                headerName="TP +1 ATR" if column.startswith("TP") else "SL -1 ATR",
                headerTooltip=(
                    "Reference take-profit: current market price plus one 14-session ATR for longs; reversed for shorts."
                    if column.startswith("TP")
                    else "Reference stop-loss: current market price minus one 14-session ATR for longs; reversed for shorts."
                ),
                type=["numericColumn"],
                valueFormatter=atr_price_formatter,
                width=126,
                minWidth=118,
            )
    for column in ["Unrealized P&L"]:
        if column in grid_frame:
            label = "U-P&L"
            builder.configure_column(
                column,
                headerName=label,
                headerTooltip=currency_header(label),
                type=["numericColumn"],
                valueFormatter=signed_money_formatter,
                cellStyle=pnl_style,
                width=136,
                minWidth=128,
            )
    if "Underlying" in grid_frame:
        builder.configure_column(
            "Underlying",
            headerName="Underlying",
            valueFormatter=diagnostic_text_formatter,
            width=124,
            minWidth=108,
        )
    for column in ["HV 5D", "HV 20D"]:
        if column in grid_frame:
            builder.configure_column(column, type=["numericColumn"], valueFormatter=percent_formatter, width=112, minWidth=102)
    if "CCI 20" in grid_frame:
        builder.configure_column("CCI 20", type=["numericColumn"], valueFormatter=number_formatter, width=105, minWidth=96)
    for column in ["BB Width", "BB 6M %ile"]:
        if column in grid_frame:
            builder.configure_column(column, type=["numericColumn"], valueFormatter=percent_formatter, width=116, minWidth=108)
    if "BB Z" in grid_frame:
        builder.configure_column("BB Z", type=["numericColumn"], valueFormatter=number_formatter, width=95, minWidth=88)
    if "Squeeze" in grid_frame:
        builder.configure_column("Squeeze", valueFormatter=diagnostic_text_formatter, width=106, minWidth=96)
    if "As Of" in grid_frame:
        builder.configure_column("As Of", headerName="As Of", width=150, minWidth=132)
    custom_css = {
        ".ag-root-wrapper": {
            "background-color": "#07101a !important",
            "border": "1px solid rgba(72, 92, 122, 0.28) !important",
            "border-radius": "10px !important",
            "box-shadow": "0 22px 50px rgba(0, 0, 0, 0.18) !important",
        },
        ".ag-root": {"background-color": "#07101a !important"},
        ".ag-root-wrapper-body": {"background-color": "#07101a !important"},
        ".ag-body": {"background-color": "#07101a !important"},
        ".ag-body-viewport": {"background-color": "#07101a !important"},
        ".ag-body-horizontal-scroll": {"background-color": "#07101a !important"},
        ".ag-body-horizontal-scroll-viewport": {"background-color": "#07101a !important"},
        ".ag-center-cols-clipper": {"background-color": "#07101a !important"},
        ".ag-center-cols-viewport": {"background-color": "#07101a !important"},
        ".ag-center-cols-container": {"background-color": "#07101a !important"},
        ".ag-pinned-left-cols-container": {"background-color": "#07101a !important"},
        ".ag-horizontal-left-spacer": {"background-color": "#07101a !important"},
        ".ag-horizontal-right-spacer": {"background-color": "#07101a !important"},
        ".ag-header": {
            "background-color": "#0f1826 !important",
            "border-bottom": "1px solid rgba(93, 113, 143, 0.30) !important",
        },
        ".ag-header-cell": {
            "background-color": "#0f1826 !important",
            "border-right": "1px solid rgba(65, 84, 112, 0.28) !important",
            # AG Grid positions headers and cells with the same pixel widths.
            # Streamlit's inherited content-box rule otherwise adds our padding
            # to body cells only, shifting every row progressively to the right.
            "box-sizing": "border-box !important",
            "padding-left": "10px !important",
            "padding-right": "10px !important",
        },
        ".ag-header-cell-label": {
            "color": "#8fa2bd !important",
            "font-weight": "800 !important",
            "letter-spacing": "0.04em !important",
            "line-height": "1.15 !important",
            "white-space": "nowrap !important",
            "overflow": "hidden !important",
            "text-overflow": "ellipsis !important",
        },
        ".ag-header-cell-text": {
            "overflow": "hidden !important",
            "text-overflow": "ellipsis !important",
            "white-space": "nowrap !important",
        },
        ".ag-row": {
            "background-color": "#08101a !important",
            "border-bottom": "1px solid rgba(65, 84, 112, 0.16) !important",
        },
        ".ag-row-odd": {"background-color": "#0c1521 !important"},
        ".ag-row-hover": {"background-color": "#152336 !important"},
        ".ag-cell": {
            "color": "#dbeafe !important",
            "border-color": "rgba(65, 84, 112, 0.16) !important",
            "box-sizing": "border-box !important",
            "font-size": "0.84rem !important",
            "line-height": "38px !important",
            "padding-left": "10px !important",
            "padding-right": "10px !important",
            "overflow": "hidden !important",
            "white-space": "nowrap !important",
            "text-overflow": "ellipsis !important",
        },
        ".ag-cell-value": {
            "overflow": "hidden !important",
            "white-space": "nowrap !important",
            "text-overflow": "ellipsis !important",
        },
        ".ag-paging-panel": {
            "background-color": "#07101a !important",
            "color": "#9fb0c7 !important",
            "border-top": "1px solid rgba(65, 84, 112, 0.20) !important",
        },
    }

    response = AgGrid(
        grid_frame,
        gridOptions=builder.build(),
        height=min(640, max(360, 100 + len(grid_frame) * 38)),
        theme="dark",
        update_mode=GridUpdateMode.VALUE_CHANGED,
        update_on=["cellValueChanged"],
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        allow_unsafe_jscode=True,
        custom_css=custom_css,
        show_search=True,
        show_download_button=False,
        key=holdings_aggrid_key(grid_frame),
        server_sync_strategy="client_wins",
    )
    returned_data = getattr(response, "data", None)
    if returned_data is None and isinstance(response, dict):
        returned_data = response.get("data")
    returned_rows = (
        returned_data.to_dict("records")
        if isinstance(returned_data, pd.DataFrame)
        else returned_data if isinstance(returned_data, list) else []
    )
    current_styles = load_holding_styles(DEFAULT_HOLDING_PLAN_PATH)
    updated_styles = dict(current_styles)
    for row in returned_rows:
        plan_key = str(row.get("Plan Key") or "")
        style = str(row.get("Holding Style") or "")
        if not plan_key or style not in HOLDING_STYLES:
            continue
        if style:
            updated_styles[plan_key] = style
        else:
            updated_styles.pop(plan_key, None)
    if updated_styles != current_styles:
        save_holding_styles(updated_styles, DEFAULT_HOLDING_PLAN_PATH)
        st.toast("Holding style saved.")
        st.rerun()
    st.caption(
        f"Sort, filter, search, and resize columns inside the grid. Money columns are shown in {REPORTING_CURRENCY}; "
        "Native CCY keeps the original instrument currency. Click Style to choose Trading or Investing; the choice is saved locally. "
        "TP/SL are reference levels based on current market price ± one 14-session ATR."
    )
    return True


def format_risk_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Market Value", "Exposure Value"):
        if column in out:
            out[column] = out[column].map(money)
    if "Weight" in out:
        out["Weight"] = out["Weight"].map(percent)
    if "Unrealized P&L" in out:
        out["Unrealized P&L"] = out["Unrealized P&L"].map(signed_money)
    if "Unrealized P&L %" in out:
        out["Unrealized P&L %"] = out["Unrealized P&L %"].map(percent)
    out = out.rename(
        columns={
            "Market Value": currency_header("Market Value"),
            "Exposure Value": currency_header("Exposure Value"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
        }
    )
    return out


def cash_currency_for_holdings(frame: pd.DataFrame) -> str:
    if frame.empty or "Currency" not in frame:
        return "Cash"
    currencies = sorted(
        item
        for item in frame["Currency"].dropna().astype(str).str.upper().unique().tolist()
        if item
    )
    if len(currencies) == 1:
        return currencies[0]
    if "USD" in currencies:
        return "USD"
    return "Cash"


def cash_not_already_in_holdings(frame: pd.DataFrame, cash_value: float | None) -> float | None:
    """Return the account cash amount not already represented as holdings rows."""

    if cash_value is None:
        return None
    cash_number = _float(cash_value)
    if cash_number is None:
        return None
    if frame.empty or "Asset Class" not in frame or "Market Value" not in frame:
        return cash_number
    asset_class = frame["Asset Class"].fillna("").astype(str).str.lower()
    cash_rows = frame.loc[asset_class.str.contains("cash", na=False)].copy()
    if cash_rows.empty:
        return cash_number
    represented_cash = pd.to_numeric(cash_rows["Market Value"], errors="coerce").sum(min_count=1)
    represented_cash_value = 0.0 if pd.isna(represented_cash) else float(represented_cash)
    return cash_number - represented_cash_value


def clean_symbol_text(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "missing", "n/a", "na", "<na>"}:
        return ""
    return text.upper()


def cash_like_holdings_value(frame: pd.DataFrame) -> float:
    if frame.empty or "Symbol" not in frame or "Market Value" not in frame:
        return 0.0
    symbols = frame["Symbol"].map(clean_symbol_text)
    cash_like_symbols = set(CASH_EQUIVALENT_SYMBOLS) | {"XEON.DE"}
    values = pd.to_numeric(
        frame.loc[symbols.isin(cash_like_symbols), "Market Value"],
        errors="coerce",
    )
    total = values.sum(min_count=1)
    return 0.0 if pd.isna(total) else float(total)


def cash_with_equivalents(cash_value: float | None, cash_like_value: float) -> float | None:
    cash_number = _float(cash_value)
    if cash_number is None and cash_like_value == 0:
        return None
    return (cash_number or 0.0) + cash_like_value


def clean_underlying_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "Symbol" not in frame:
        return frame
    out = frame.copy()
    out["Symbol"] = out["Symbol"].map(clean_symbol_text)
    return out.loc[out["Symbol"].astype(bool)].reset_index(drop=True)


def concentration_value(frame: pd.DataFrame, metric: str) -> Any:
    if frame.empty or "Metric" not in frame:
        return None
    rows = frame.loc[frame["Metric"].astype(str).eq(metric)]
    if rows.empty:
        return None
    return rows.iloc[0].get("Value")


def concentration_detail(frame: pd.DataFrame, metric: str) -> str:
    if frame.empty or "Metric" not in frame:
        return "missing"
    rows = frame.loc[frame["Metric"].astype(str).eq(metric)]
    if rows.empty:
        return "missing"
    value = str(rows.iloc[0].get("Detail") or "").strip()
    return value or "missing"


def format_exposure_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in (
        "Exposure Value",
        "Market Value",
        "Direct Market Value",
        "Option Market Value",
        "Delta Dollars",
    ):
        if column in out:
            out[column] = out[column].map(money)
    for column in ("Weight", "HV 5D", "HV 20D"):
        if column in out:
            out[column] = out[column].map(percent)
    if "Unrealized P&L" in out:
        out["Unrealized P&L"] = out["Unrealized P&L"].map(signed_money)
    if "Net Option Delta" in out:
        out["Net Option Delta"] = out["Net Option Delta"].map(lambda value: decimal(value, 2))
    for column in ("Option Contract Shares", "Gross Option Contract Shares"):
        if column in out:
            out[column] = out[column].map(quantity)
    if "Rows" in out:
        out["Rows"] = out["Rows"].map(quantity)
    out = out.rename(
        columns={
            "Exposure Value": currency_header("Exposure Value"),
            "Market Value": currency_header("Market Value"),
            "Direct Market Value": currency_header("Direct Market Value"),
            "Option Market Value": currency_header("Option Market Value"),
            "Delta Dollars": currency_header("Delta Dollars"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
        }
    )
    return out


def option_adjusted_underlying_frame(underlying_exposure: pd.DataFrame, *, nav: float | None = None) -> pd.DataFrame:
    """Return an underlying-level frame for option-aware risk charts.

    Market value is still the accounting mark. Economic exposure uses delta
    dollars when available; otherwise it falls back to marked market value, with
    gross option contract shares shown separately in the table.
    """

    columns = [
        "Symbol",
        "Market Value",
        "Economic Exposure",
        "Weight",
        "Direct Market Value",
        "Option Market Value",
        "Option Cost Basis",
        "Option Contract Shares",
        "Gross Option Contract Shares",
        "Net Option Delta",
        "Delta Dollars",
        "Unrealized P&L",
        "Rows",
    ]
    if underlying_exposure.empty:
        return pd.DataFrame(columns=columns)

    out = underlying_exposure.copy()
    out["Symbol"] = out["Underlying"].astype(str)
    for column in (
        "Market Value",
        "Direct Market Value",
        "Option Market Value",
        "Option Cost Basis",
        "Delta Dollars",
        "Unrealized P&L",
    ):
        if column not in out:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce")
    delta_abs = out["Delta Dollars"].abs()
    market_abs = out["Market Value"].abs()
    out["Economic Exposure"] = delta_abs.where(delta_abs.notna() & delta_abs.gt(0), market_abs)

    denominator = None
    if nav is not None and pd.notna(nav) and abs(float(nav)) > 0:
        denominator = abs(float(nav))
    if denominator is None:
        total = out["Economic Exposure"].sum()
        denominator = None if pd.isna(total) or total <= 0 else float(total)
    out["Weight"] = 0.0 if denominator is None else out["Economic Exposure"] / denominator
    return (
        out.reindex(columns=columns)
        .sort_values("Economic Exposure", ascending=False)
        .reset_index(drop=True)
    )


def format_option_adjusted_underlying_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in (
        "Market Value",
        "Economic Exposure",
        "Direct Market Value",
        "Option Market Value",
        "Delta Dollars",
        "Unrealized P&L",
    ):
        if column in out:
            formatter = signed_money if column == "Unrealized P&L" else money
            out[column] = out[column].map(formatter)
    if "Weight" in out:
        out["Weight"] = out["Weight"].map(percent)
    for column in ("Rows", "Option Contract Shares", "Gross Option Contract Shares"):
        if column in out:
            out[column] = out[column].map(quantity)
    if "Net Option Delta" in out:
        out["Net Option Delta"] = out["Net Option Delta"].map(lambda value: decimal(value, 2))
    return out.rename(
        columns={
            "Market Value": currency_header("Market Value"),
            "Economic Exposure": currency_header("Economic Exposure"),
            "Direct Market Value": currency_header("Direct Market Value"),
            "Option Market Value": currency_header("Option Market Value"),
            "Option Cost Basis": currency_header("Option Cost Basis"),
            "Delta Dollars": currency_header("Delta Dollars"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
        }
    )


def format_concentration_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()

    def _format_row(row: pd.Series) -> str:
        metric = str(row.get("Metric") or "")
        value = row.get("Value")
        if metric in {"Largest exposure", "Largest non-cash", "Top 3 weight", "Top 5 weight", "Top 10 weight"}:
            return percent(value)
        if metric == "HHI":
            return decimal(value, 3)
        if metric == "Effective positions":
            return decimal(value, 1)
        if metric == "Position rows":
            return quantity(value)
        return decimal(value, 2)

    out["Value"] = out.apply(_format_row, axis=1)
    return out


def format_concentration_curve_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "Exposure Value" in out:
        out["Exposure Value"] = out["Exposure Value"].map(money)
    for column in ("Weight", "Cumulative Weight"):
        if column in out:
            out[column] = out[column].map(percent)
    out = out.rename(columns={"Exposure Value": currency_header("Exposure Value")})
    return out


def risk_lab_cache_symbols(symbols: list[str]) -> tuple[str, ...]:
    expanded: list[str] = ["QQQ", "SPY"]
    for symbol in symbols:
        text = str(symbol or "").upper().strip()
        if not text:
            continue
        if text == "TENCENT":
            expanded.extend(["TENCENT", "TCEHY", "0700.HK", "700.HK"])
        elif text == "BRK B":
            expanded.extend(["BRK B", "BRK.B", "BRK-B"])
        else:
            expanded.append(text)
            alias = PORTFOLIO_TICKER_ALIASES.get(text)
            if alias:
                expanded.append(str(alias).upper().strip())
    return tuple(dict.fromkeys(expanded))


def live_price_history_symbols(positions: pd.DataFrame) -> tuple[str, ...]:
    """Return all symbols needed for holding-level HV plus risk-lab history."""

    symbols: list[str] = []
    for column in ("symbol", "Symbol", "underlying", "Underlying"):
        if column in positions:
            symbols.extend(positions[column].dropna().astype(str).str.upper().str.strip().tolist())

    exposure = underlying_exposure_report(positions)
    if not exposure.empty and "Underlying" in exposure:
        symbols.extend(exposure["Underlying"].dropna().astype(str).str.upper().str.strip().tolist())

    return risk_lab_cache_symbols([symbol for symbol in dict.fromkeys(symbols) if symbol])


def holding_atr_symbol_map(positions: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    """Map held equity symbols to the symbols requested from FMP."""

    if positions.empty:
        return ()
    symbol_column = "symbol" if "symbol" in positions else "Symbol"
    asset_column = "asset_class" if "asset_class" in positions else "Asset Class"
    if symbol_column not in positions or asset_column not in positions:
        return ()
    pairs: list[tuple[str, str]] = []
    for row in positions[[symbol_column, asset_column]].to_dict("records"):
        asset_class = str(row.get(asset_column) or "").lower()
        if not any(token in asset_class for token in ("equity", "stock", "etf")):
            continue
        holding_symbol = str(row.get(symbol_column) or "").upper().strip()
        if not holding_symbol:
            continue
        provider_symbol = str(PORTFOLIO_TICKER_ALIASES.get(holding_symbol, holding_symbol)).upper().strip()
        pairs.append((holding_symbol, provider_symbol))
    return tuple(dict.fromkeys(pairs))


@st.cache_data(ttl=21_600, show_spinner=False)
def load_holding_atr_history(
    symbol_map: tuple[tuple[str, str], ...],
    fmp_api_key: str | None,
) -> pd.DataFrame:
    """Return FMP-first daily OHLC history used only for holding ATR levels."""

    columns = ["symbol", "date", "open", "high", "low", "close"]
    if not symbol_map:
        return pd.DataFrame(columns=columns)

    provider_symbols = tuple(dict.fromkeys(provider for _, provider in symbol_map))
    if fmp_api_key:
        status = market_cache_status(provider_symbols, source="fmp", max_age_hours=18.0)
        refresh_symbols = tuple(
            status.loc[status["State"].isin(["missing", "stale"]), "Symbol"].astype(str)
        )
        if refresh_symbols:
            # FMP is the primary provider. Individual vendor failures are
            # contained by refresh_fmp_market_cache and handled by cache fallback.
            refresh_fmp_market_cache(refresh_symbols, api_key=fmp_api_key, period="6mo")

    rows: list[pd.DataFrame] = []
    for holding_symbol, provider_symbol in symbol_map:
        history = load_cached_market_history(
            provider_symbol,
            source="fmp",
            lookback_days=180,
        )
        if history.empty:
            # Keep existing Yahoo cache as a last-resort fallback; no Yahoo
            # network request is made from this holdings path.
            history = load_cached_market_history(
                provider_symbol,
                source="yahoo",
                lookback_days=180,
            )
        if history.empty:
            continue
        normalized = history.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
            }
        )
        normalized["symbol"] = holding_symbol
        rows.append(normalized.reindex(columns=columns))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def volatility_with_aliases(volatility: pd.DataFrame) -> pd.DataFrame:
    """Duplicate HV rows across broker/vendor aliases used by holdings."""

    if volatility.empty or "symbol" not in volatility:
        return volatility
    rows = volatility.copy()
    rows["symbol"] = rows["symbol"].astype(str).str.upper().str.strip()
    alias_groups = [
        ("TENCENT", "TCEHY", "0700.HK", "700.HK"),
        ("BRK B", "BRK.B", "BRK-B"),
    ]
    alias_groups.extend((key.upper(), value.upper()) for key, value in PORTFOLIO_TICKER_ALIASES.items())

    additions: list[pd.Series] = []
    present = set(rows["symbol"].tolist())
    for group in alias_groups:
        source = next((symbol for symbol in group if symbol in present), "")
        if not source:
            continue
        source_rows = rows.loc[rows["symbol"].eq(source)]
        if source_rows.empty:
            continue
        source_row = source_rows.iloc[-1]
        for alias in group:
            if alias in present:
                continue
            duplicate = source_row.copy()
            duplicate["symbol"] = alias
            additions.append(duplicate)
            present.add(alias)

    if additions:
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
    return rows.drop_duplicates(subset=["symbol"], keep="last").reset_index(drop=True)


def account_wrapper_selector() -> str:
    """Top-level wrapper for live portfolio account views."""

    st.caption("Account wrapper")
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Account wrapper",
            ACCOUNT_SCOPES,
            default=st.session_state.get("live_account_scope", US_ACCOUNT_SCOPE),
            label_visibility="collapsed",
            key="live_account_scope",
        )
        return str(selected or US_ACCOUNT_SCOPE)
    return st.radio(
        "Account wrapper",
        ACCOUNT_SCOPES,
        horizontal=True,
        label_visibility="collapsed",
        key="live_account_scope",
    )


def cn_account_mask(frame: pd.DataFrame) -> pd.Series:
    """Detect rows belonging to the planned 华源证券/QMT CN account wrapper."""

    if frame.empty:
        return pd.Series(False, index=frame.index)
    text_parts = []
    for column in (
        "Broker",
        "broker",
        "Profile",
        "profile",
        "Account",
        "account_id",
        "Asset Class",
        "asset_class",
        "Currency",
        "currency",
        "Native Currency",
        "native_currency",
        "Symbol",
        "symbol",
        "metadata_json",
    ):
        if column in frame:
            text_parts.append(frame[column].fillna("").astype(str))
    if not text_parts:
        return pd.Series(False, index=frame.index)
    text = text_parts[0]
    for part in text_parts[1:]:
        text = text + " " + part
    normalized = text.str.upper()
    return normalized.str.contains(
        r"QMT|HUAYUAN|华源|EQUITY_CN|OPTIONS_CN|FUTURES_CN|\.SH|\.SZ|CNY|CNH",
        regex=True,
        na=False,
    )


def holdings_for_account_scope(holdings: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Filter enriched holdings to the selected account wrapper."""

    if holdings.empty:
        return holdings
    mask = cn_account_mask(holdings)
    if scope == CN_ACCOUNT_SCOPE:
        return holdings.loc[mask].reset_index(drop=True)
    return holdings.loc[~mask].reset_index(drop=True)


def positions_for_account_scope(
    positions: pd.DataFrame,
    all_positions: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    """Filter raw positions to the selected account wrapper."""

    frame = all_positions if scope == CN_ACCOUNT_SCOPE else positions
    if frame.empty:
        return frame
    mask = cn_account_mask(frame)
    out = frame.loc[mask] if scope == CN_ACCOUNT_SCOPE else frame.loc[~mask]
    if "environment" in out:
        out = out.loc[out["environment"].astype(str).str.lower().eq("live")]
    return out.reset_index(drop=True)


def nav_from_holdings(holdings: pd.DataFrame) -> float | None:
    if holdings.empty or "Market Value" not in holdings:
        return None
    values = pd.to_numeric(holdings["Market Value"], errors="coerce").abs()
    total = values.sum(min_count=1)
    return None if pd.isna(total) or total == 0 else float(total)


def cash_from_holdings(holdings: pd.DataFrame) -> float | None:
    if holdings.empty or "Market Value" not in holdings:
        return None
    asset = holdings.get("Asset Class", pd.Series("", index=holdings.index)).astype(str).str.lower()
    symbol = holdings.get("Symbol", pd.Series("", index=holdings.index)).astype(str).str.upper()
    mask = asset.eq("cash") | symbol.str.contains("CASH", na=False)
    values = pd.to_numeric(holdings.loc[mask, "Market Value"], errors="coerce")
    total = values.sum(min_count=1)
    return None if pd.isna(total) else float(total)


def pnl_from_holdings(holdings: pd.DataFrame) -> float | None:
    if holdings.empty or "Unrealized P&L" not in holdings:
        return None
    values = pd.to_numeric(holdings["Unrealized P&L"], errors="coerce")
    total = values.sum(min_count=1)
    return None if pd.isna(total) else float(total)


def format_factor_lab_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()

    def _format(row: pd.Series) -> str:
        metric = str(row.get("Metric") or "")
        value = row.get("Value")
        if metric in {"R squared", "Portfolio vol", "Residual vol"}:
            return percent(value)
        if metric in {"Observations", "Portfolio assets covered", "Factors available"}:
            return quantity(value)
        return decimal(value, 2)

    out["Value"] = out.apply(_format, axis=1)
    return out


def format_factor_betas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["Beta"] = out["Beta"].map(lambda value: decimal(value, 3))
    return out


def format_pca_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()

    def _format(row: pd.Series) -> str:
        metric = str(row.get("Metric") or "")
        value = row.get("Value")
        if metric == "PC1 variance":
            return percent(value)
        if metric in {"Observations", "Assets covered", "Components to 80%"}:
            return quantity(value)
        return decimal(value, 2)

    out["Value"] = out.apply(_format, axis=1)
    return out


def format_pca_spectrum(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Explained Variance", "Cumulative"):
        if column in out:
            out[column] = out[column].map(percent)
    if "Eigenvalue" in out:
        out["Eigenvalue"] = out["Eigenvalue"].map(lambda value: decimal(value, 3))
    return out


def format_pca_drivers(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "Explained Variance" in out:
        out["Explained Variance"] = out["Explained Variance"].map(percent)
    return out


def render_factor_beta_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty or "Factor" not in frame or "Beta" not in frame:
        st.info("No factor betas are available yet.")
        return
    chart = frame.copy()
    chart["Beta"] = pd.to_numeric(chart["Beta"], errors="coerce")
    chart = chart.dropna(subset=["Beta"]).sort_values("Beta")
    if chart.empty:
        st.info("No factor betas are available yet.")
        return
    colors = ["#fb7185" if value < 0 else "#5eead4" for value in chart["Beta"]]
    fig = go.Figure(
        go.Bar(
            y=chart["Factor"],
            x=chart["Beta"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate="<b>%{y}</b><br>Beta: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="rgba(148,163,184,0.35)", line_width=1)
    style_dark_plotly(
        fig,
        height=300,
        xaxis_title="Beta",
        margin=dict(t=10, b=42, l=150, r=24),
        hovermode=None,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def render_pca_spectrum_chart(frame: pd.DataFrame, *, key: str) -> None:
    if frame.empty or "Component" not in frame or "Explained Variance" not in frame:
        st.info("No PCA spectrum is available yet.")
        return
    chart = frame.head(8).copy()
    chart["Explained Variance"] = pd.to_numeric(chart["Explained Variance"], errors="coerce").fillna(0.0)
    chart["Cumulative"] = pd.to_numeric(chart.get("Cumulative", pd.Series(0.0, index=chart.index)), errors="coerce").fillna(0.0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart["Component"],
            y=chart["Explained Variance"] * 100,
            name="Explained",
            marker={"color": "#5eead4", "line": {"width": 0}},
            hovertemplate="<b>%{x}</b><br>Explained: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart["Component"],
            y=chart["Cumulative"] * 100,
            name="Cumulative",
            mode="lines+markers",
            line={"color": "#60a5fa", "width": 2},
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
        )
    )
    style_dark_plotly(
        fig,
        height=300,
        yaxis_title="Variance %",
        margin=dict(t=10, b=42, l=52, r=24),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def render_horizontal_exposure_bar(
    frame: pd.DataFrame,
    *,
    label_column: str,
    value_column: str = "Exposure Value",
    weight_column: str = "Weight",
    height: int = 340,
    key: str,
) -> None:
    if frame.empty or label_column not in frame or value_column not in frame:
        st.info("No exposure rows are available.")
        return
    chart = frame.copy()
    chart[value_column] = pd.to_numeric(chart[value_column], errors="coerce").fillna(0.0)
    chart = chart.loc[chart[value_column] > 0].sort_values(value_column).tail(14)
    if chart.empty:
        st.info("No positive exposure values are available.")
        return
    has_weights = weight_column in chart
    if has_weights:
        weights = pd.to_numeric(chart[weight_column], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(0.0, index=chart.index)
    fig = go.Figure(
        go.Bar(
            y=chart[label_column],
            x=chart[value_column],
            orientation="h",
            marker={"color": "#5eead4", "line": {"width": 0}},
            text=[percent(value) for value in weights] if has_weights else None,
            textposition="outside" if has_weights else None,
            hovertemplate="<b>%{y}</b><br>Exposure: %{x:,.2f}<extra></extra>",
        )
    )
    style_dark_plotly(
        fig,
        height=height,
        xaxis_title="Exposure Value",
        margin=dict(t=12, b=68, l=120, r=32),
        hovermode=None,
    )
    fig.update_xaxes(title_standoff=14, automargin=True)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def render_concentration_pareto(frame: pd.DataFrame, *, key: str = "live_concentration_pareto") -> None:
    if frame.empty:
        st.info("No concentration curve is available.")
        return
    chart = frame.head(16).copy()
    chart["Weight"] = pd.to_numeric(chart["Weight"], errors="coerce").fillna(0.0)
    chart["Cumulative Weight"] = pd.to_numeric(chart["Cumulative Weight"], errors="coerce").fillna(0.0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart["Symbol"],
            y=chart["Weight"] * 100,
            name="Position weight",
            marker={"color": "#5eead4", "line": {"width": 0}},
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart["Symbol"],
            y=chart["Cumulative Weight"] * 100,
            name="Cumulative",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#60a5fa", "width": 2.6},
            marker={"size": 7},
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.2f}%<extra></extra>",
        )
    )
    style_dark_plotly(
        fig,
        height=360,
        xaxis_title="Position",
        yaxis_title="Weight %",
        margin=dict(t=14, b=48, l=48, r=52),
    )
    max_cumulative = max(100.0, float((chart["Cumulative Weight"] * 100).max()))
    fig.update_layout(
        yaxis2={
            "title": "Cumulative %",
            "overlaying": "y",
            "side": "right",
            "range": [0, max_cumulative * 1.08],
            "showgrid": False,
            "tickfont": {"color": "#94A3B8"},
            "title_font": {"color": "#94A3B8"},
        },
        legend={"orientation": "h", "y": 1.08, "x": 0.0},
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def render_exposure_treemap(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No position exposure rows are available.")
        return

    chart = frame.loc[pd.to_numeric(frame["Exposure Value"], errors="coerce") > 0].copy()
    if chart.empty:
        st.info("No positive exposure values are available.")
        return

    labels = ["Portfolio"]
    parents = [""]
    ids = ["Portfolio"]
    values = [float(chart["Exposure Value"].sum())]
    colors = [0.0]
    customdata = [["Portfolio", 1.0, values[0], 0.0, 0.0]]

    sleeve_summary = (
        chart.groupby("Sleeve", sort=False)
        .agg(
            exposure=("Exposure Value", "sum"),
            weight=("Weight", "sum"),
            pnl=("Unrealized P&L", "sum"),
        )
        .reset_index()
    )
    sleeve_summary["pnl_pct"] = sleeve_summary["pnl"] / sleeve_summary["exposure"].replace(0, pd.NA)
    sleeve_summary["pnl_pct"] = sleeve_summary["pnl_pct"].fillna(0.0)

    for _, row in sleeve_summary.iterrows():
        sleeve = str(row["Sleeve"])
        labels.append(sleeve)
        parents.append("Portfolio")
        ids.append(f"sleeve::{sleeve}")
        values.append(float(row["exposure"]))
        colors.append(float(row["pnl_pct"]))
        customdata.append([sleeve, float(row["weight"]), float(row["exposure"]), float(row["pnl"]), float(row["pnl_pct"])])

    for index, row in chart.reset_index(drop=True).iterrows():
        symbol = str(row["Symbol"])
        sleeve = str(row["Sleeve"])
        labels.append(symbol)
        parents.append(f"sleeve::{sleeve}")
        ids.append(f"position::{sleeve}::{symbol}::{index}")
        values.append(float(row["Exposure Value"]))
        colors.append(float(row["Unrealized P&L %"]))
        customdata.append(
            [
                sleeve,
                float(row["Weight"]),
                float(row["Exposure Value"]),
                float(row["Unrealized P&L"]),
                float(row["Unrealized P&L %"]),
            ]
        )

    color_abs = max(abs(value) for value in colors if value is not None) if colors else 0.0
    color_abs = max(color_abs, 0.05)
    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            ids=ids,
            values=values,
            branchvalues="total",
            marker={
                "colors": colors,
                "colorscale": [[0, "#FB7185"], [0.5, "#475569"], [1, "#2DD4BF"]],
                "cmid": 0,
                "cmin": -color_abs,
                "cmax": color_abs,
                "line": {"color": "rgba(8,12,18,0.92)", "width": 2},
            },
            customdata=customdata,
            texttemplate="<b>%{label}</b><br>%{percentParent:.1%}",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Sleeve: %{customdata[0]}<br>"
                "Exposure: %{customdata[2]:,.2f}<br>"
                "Weight: %{customdata[1]:.2%}<br>"
                "Unrealized P&L: %{customdata[3]:+,.2f}<br>"
                "P&L %: %{customdata[4]:+.2%}<extra></extra>"
            ),
        )
    )
    style_dark_plotly(fig, height=390, margin=dict(t=8, b=8, l=8, r=8), hovermode=None)
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True, theme=None, key="live_exposure_treemap")


def render_position_risk_scatter(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No position risk rows are available.")
        return

    chart = frame.loc[(frame["Symbol"].astype(str) != "Cash") & (frame["Exposure Value"] > 0)].copy()
    if chart.empty:
        st.info("No non-cash position risk rows are available.")
        return

    max_exposure = float(chart["Exposure Value"].max())
    fig = go.Figure()
    palette = {
        "Equity (index)": "#60A5FA",
        "Equity (defensive)": "#22C55E",
        "Equity (core)": "#A78BFA",
        "Equity (aggressive)": "#F59E0B",
        "Options": "#FB7185",
        "Fixed income": "#2DD4BF",
        "Commodity": "#F97316",
        "Other": "#94A3B8",
    }
    for sleeve, group in chart.groupby("Sleeve", sort=False):
        sizes = 14 + 34 * (group["Exposure Value"] / max_exposure).pow(0.5)
        fig.add_trace(
            go.Scatter(
                x=group["Weight"] * 100,
                y=group["Unrealized P&L %"] * 100,
                mode="markers+text",
                text=group["Symbol"],
                textposition="top center",
                name=str(sleeve),
                marker={
                    "size": sizes,
                    "color": palette.get(str(sleeve), "#94A3B8"),
                    "opacity": 0.78,
                    "line": {"color": "rgba(248,250,252,0.22)", "width": 1},
                },
                customdata=group[["Exposure Value", "Unrealized P&L", "Market Value"]].to_numpy(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Weight: %{x:.2f}%<br>"
                    "Unrealized P&L %: %{y:+.2f}%<br>"
                    "Exposure: %{customdata[0]:,.2f}<br>"
                    "Unrealized P&L: %{customdata[1]:+,.2f}<br>"
                    "Market Value: %{customdata[2]:,.2f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.30)", line_dash="dot")
    fig.add_vline(x=5, line_color="rgba(245,158,11,0.26)", line_dash="dot")
    style_dark_plotly(
        fig,
        height=390,
        xaxis_title="Portfolio Weight %",
        yaxis_title="Unrealized P&L %",
        margin=dict(t=12, b=42, l=50, r=20),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None, key="live_position_risk_scatter")


def winners_losers_tables(frame: pd.DataFrame, *, limit: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["Symbol", "Sleeve", "Weight", "Market Value", "Unrealized P&L", "Unrealized P&L %"]
    if frame.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    candidates = frame.loc[frame["Symbol"].astype(str) != "Cash"].copy()
    if candidates.empty:
        empty = pd.DataFrame(columns=columns)
        return empty, empty
    winners = candidates.sort_values("Unrealized P&L", ascending=False).head(limit).reindex(columns=columns)
    losers = candidates.sort_values("Unrealized P&L", ascending=True).head(limit).reindex(columns=columns)
    return format_risk_table(winners), format_risk_table(losers)


def load_live_portfolio_data() -> dict[str, Any]:
    settings = load_settings()
    account_ledger = default_account_ledger_path()
    portfolio_nav = load_historical_nav(default_portfolio_ledger_path())
    manual_sync_rows = sync_manual_external_positions_from_json(account_ledger)
    unified_write = materialize_unified_live_account_snapshot(account_ledger)
    latest_nav = load_latest_account_nav(
        account_ledger,
        environment="live",
        profile=UNIFIED_LIVE_PROFILE,
    )
    unified_nav_raw = load_account_nav_history(
        account_ledger,
        environment="live",
        profile=UNIFIED_LIVE_PROFILE,
    )
    broker_nav_raw = load_account_nav_history(
        account_ledger,
        environment="live",
        profile=DEFAULT_LIVE_BROKER_PROFILE,
    )
    nav_raw = blended_live_nav_history(
        unified_nav_raw,
        broker_nav_raw,
        manual_usd_value=unified_write.manual_usd_value,
        manual_usd_cash=unified_write.manual_usd_cash,
    )
    nav_history = account_nav_drawdowns(nav_raw)
    broker_positions = load_latest_account_positions(
        account_ledger,
        environment="live",
        profile=DEFAULT_LIVE_BROKER_PROFILE,
    )
    manual_positions = load_manual_external_positions_as_account_positions(
        account_ledger,
        environment="live",
    )
    positions = load_latest_account_positions(
        account_ledger,
        environment="live",
        profile=UNIFIED_LIVE_PROFILE,
    )
    all_live_positions = load_latest_account_positions(
        account_ledger,
        environment="live",
    )
    position_history = load_account_position_history(
        account_ledger,
        environment="live",
        profile=UNIFIED_LIVE_PROFILE,
    )
    events = load_account_trade_events(account_ledger, environment="live", limit=100)
    static_price_history = load_price_history()
    history_symbols = live_price_history_symbols(positions)
    cached_yahoo_history = load_cached_price_history(
        history_symbols,
        source="yahoo",
        lookback_days=900,
    )
    cached_fmp_history = load_cached_price_history(
        history_symbols,
        source="fmp",
        lookback_days=900,
    )
    atr_price_history = load_holding_atr_history(
        holding_atr_symbol_map(positions),
        settings.fmp_api_key,
    )
    # FMP is the primary cache provider. Passing it last makes it win same-day
    # duplicates while Yahoo remains available as a gap-filling fallback.
    price_history = combine_price_histories(static_price_history, cached_yahoo_history, cached_fmp_history)
    hv_source_history = pd.concat(
        [frame for frame in (static_price_history, cached_yahoo_history, cached_fmp_history) if not frame.empty],
        ignore_index=True,
    ) if not static_price_history.empty or not cached_yahoo_history.empty or not cached_fmp_history.empty else pd.DataFrame(columns=["symbol", "date", "close"])
    hv = volatility_with_aliases(historical_volatility_frame(hv_source_history, windows=(5, 20)))
    cci_rows = []
    for symbol in history_symbols:
        history = load_cached_market_history(symbol, lookback_days=220)
        cci_rows.append(
            {
                "symbol": symbol,
                "cci_20d": commodity_channel_index(history, window=20),
                **bollinger_squeeze_metrics(history, window=20, lookback=126),
            }
        )
    cci = volatility_with_aliases(pd.DataFrame(cci_rows))
    if not cci.empty:
        hv = hv.merge(cci, on="symbol", how="outer") if not hv.empty else cci
    ops_snapshot = collect_ops_status(settings=settings)
    server_sync = read_json(REPO_ROOT / "runtime" / "state" / "server_sync" / "status.json")
    return {
        "account_ledger": account_ledger,
        "manual_external_json": DEFAULT_MANUAL_EXTERNAL_INPUT_PATH,
        "manual_sync_rows": manual_sync_rows,
        "unified_write": unified_write.to_dict(),
        "latest_nav": latest_nav,
        "nav_raw": nav_raw,
        "unified_nav_raw": unified_nav_raw,
        "broker_nav_raw": broker_nav_raw,
        "nav_history": nav_history,
        "portfolio_nav": portfolio_nav,
        "broker_positions": broker_positions,
        "manual_positions": manual_positions,
        "positions": positions,
        "all_live_positions": all_live_positions,
        "position_history": position_history,
        "events": events,
        "price_history": price_history,
        "atr_price_history": atr_price_history,
        "hv": hv,
        "settings": settings,
        "ops_snapshot": ops_snapshot,
        "server_sync": server_sync,
    }


def latest_nav_value(latest_nav: pd.DataFrame, column: str) -> float | None:
    if latest_nav.empty:
        return None
    return _float(latest_nav.iloc[0].get(column))


def latest_nav_currency(latest_nav: pd.DataFrame, fallback: Any = None) -> str:
    if not latest_nav.empty and "currency" in latest_nav:
        value = str(latest_nav.iloc[0].get("currency") or "").upper().strip()
        if value:
            return value
    value = str(fallback or "").upper().strip()
    return value or "USD"


def concentration_frame(holdings: pd.DataFrame, nav: float | None) -> pd.DataFrame:
    columns = ["Symbol", "Asset Class", "Market Value", "Weight", "Unrealized P&L", "HV 5D", "HV 20D"]
    if holdings.empty:
        return pd.DataFrame(columns=columns)
    out = holdings.copy()
    out["abs_market_value"] = pd.to_numeric(out["Market Value"], errors="coerce").abs()
    out["Weight"] = None if nav in (None, 0) else out["abs_market_value"] / float(nav)
    return (
        out.sort_values("abs_market_value", ascending=False)
        .reindex(columns=columns)
        .head(20)
    )


def monthly_return_table(nav_history: pd.DataFrame) -> pd.DataFrame:
    columns = ["Month", "Return", "P&L", "Ending NAV"]
    if nav_history.empty:
        return pd.DataFrame(columns=columns)
    out = nav_history.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["month"] = out["date"].dt.to_period("M").astype(str)
    grouped = (
        out.groupby("month")
        .agg(
            start_nav=("net_liquidation", "first"),
            end_nav=("net_liquidation", "last"),
            pnl=("daily_pnl", "sum"),
        )
        .reset_index()
    )
    grouped["return"] = grouped["end_nav"] / grouped["start_nav"].replace(0, pd.NA) - 1
    return grouped.rename(
        columns={
            "month": "Month",
            "return": "Return",
            "pnl": "P&L",
            "end_nav": "Ending NAV",
        }
    ).reindex(columns=columns)


def weekly_return_table(nav_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Week",
        f"Start NAV ({REPORTING_CURRENCY})",
        f"End NAV ({REPORTING_CURRENCY})",
        f"Weekly P&L ({REPORTING_CURRENCY})",
        "Return",
        "Max Drawdown",
        "Snapshots",
    ]
    if nav_history.empty:
        return pd.DataFrame(columns=columns)

    out = nav_history.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    if out.empty:
        return pd.DataFrame(columns=columns)
    out["week_period"] = out["date"].dt.to_period("W")
    grouped = (
        out.groupby("week_period", sort=True)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            start_nav=("net_liquidation", "first"),
            end_nav=("net_liquidation", "last"),
            pnl=("daily_pnl", "sum"),
            max_drawdown=("drawdown_pct", "min"),
            snapshots=("date", "count"),
        )
        .reset_index(drop=True)
    )
    grouped["return"] = grouped["end_nav"] / grouped["start_nav"].replace(0, pd.NA) - 1
    grouped["Week"] = grouped.apply(
        lambda row: f"{pd.Timestamp(row['start_date']).strftime('%b %d')} - {pd.Timestamp(row['end_date']).strftime('%b %d')}",
        axis=1,
    )
    grouped[f"Start NAV ({REPORTING_CURRENCY})"] = grouped["start_nav"].map(money)
    grouped[f"End NAV ({REPORTING_CURRENCY})"] = grouped["end_nav"].map(money)
    grouped[f"Weekly P&L ({REPORTING_CURRENCY})"] = grouped["pnl"].map(signed_money)
    grouped["Return"] = grouped["return"].map(percent)
    grouped["Max Drawdown"] = grouped["max_drawdown"].map(percent)
    grouped["Snapshots"] = grouped["snapshots"].map(quantity)
    return grouped.tail(12).iloc[::-1].reindex(columns=columns)


def benchmark_return_series(
    price_history: pd.DataFrame,
    nav_history: pd.DataFrame,
    *,
    candidates: tuple[str, ...] = ("QQQ", "SPY"),
) -> tuple[str | None, pd.Series]:
    if price_history.empty or nav_history.empty:
        return None, pd.Series(dtype=float)
    if not {"symbol", "date", "close"}.issubset(price_history.columns):
        return None, pd.Series(dtype=float)

    dates = pd.to_datetime(nav_history["date"], errors="coerce").dropna().dt.normalize()
    if dates.empty:
        return None, pd.Series(dtype=float)

    history = price_history.copy()
    history["symbol"] = history["symbol"].astype(str).str.upper().str.strip()
    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.normalize()
    history["close"] = pd.to_numeric(history["close"], errors="coerce")
    history = history.dropna(subset=["date", "close"])

    for symbol in candidates:
        series = (
            history.loc[history["symbol"].eq(symbol.upper()), ["date", "close"]]
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .set_index("date")["close"]
        )
        if series.empty:
            continue
        aligned = series.reindex(dates).ffill()
        valid = aligned.dropna()
        if valid.empty:
            continue
        base = float(valid.iloc[0])
        if base == 0:
            continue
        returns = (aligned / base - 1.0).fillna(0.0)
        returns.index = pd.to_datetime(nav_history["date"], errors="coerce")
        return symbol.upper(), returns
    return None, pd.Series(dtype=float)


def render_live_performance_stack(
    nav_history: pd.DataFrame,
    price_history: pd.DataFrame,
    performance_summary: dict[str, Any],
) -> None:
    if nav_history.empty:
        st.info("No live NAV history has been recorded yet.")
        return

    chart = nav_history.copy()
    chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
    chart = chart.dropna(subset=["date"]).sort_values("date")
    if chart.empty:
        st.info("No live NAV history has been recorded yet.")
        return

    chart["net_liquidation"] = pd.to_numeric(chart["net_liquidation"], errors="coerce")
    chart["daily_pnl"] = pd.to_numeric(chart["daily_pnl"], errors="coerce").fillna(0.0)
    if "cumulative_return" in chart:
        cumulative = pd.to_numeric(chart["cumulative_return"], errors="coerce").fillna(0.0) * 100
    else:
        base = chart["net_liquidation"].replace(0, pd.NA).dropna()
        cumulative = pd.Series(0.0, index=chart.index) if base.empty else (chart["net_liquidation"] / float(base.iloc[0]) - 1.0).fillna(0.0) * 100
    drawdown = pd.to_numeric(chart.get("drawdown_pct", pd.Series(0.0, index=chart.index)), errors="coerce").fillna(0.0) * 100

    current_gross = _float(performance_summary.get("gross_exposure_pct"))
    if "gross_exposure_pct" in chart:
        leverage = pd.to_numeric(chart["gross_exposure_pct"], errors="coerce")
    else:
        leverage = pd.Series(current_gross, index=chart.index)
    leverage = leverage.fillna(current_gross if current_gross is not None else 0.0) * 100

    benchmark_label, benchmark = benchmark_return_series(price_history, chart)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.48, 0.27, 0.25],
    )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=cumulative,
            mode="lines+markers" if len(chart) < 3 else "lines",
            name="Portfolio",
            line={"color": "#5eead4", "width": 2.6},
            hovertemplate="Portfolio: %{y:.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if benchmark_label and not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=chart["date"],
                y=benchmark * 100,
                mode="lines+markers" if len(chart) < 3 else "lines",
                name=benchmark_label,
                line={"color": "#f59e0b", "width": 2.2, "dash": "dot"},
                hovertemplate=f"{benchmark_label}: " + "%{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"color": "#fb7185", "width": 1.7},
            fill="tozeroy",
            fillcolor="rgba(251,113,133,0.26)",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=chart["date"],
            y=leverage,
            mode="lines",
            name="Gross Exposure / NAV",
            line={"color": "#60a5fa", "width": 1.8},
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.18)",
            hovertemplate="Gross / NAV: %{y:.2f}%<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.26)", line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(148,163,184,0.26)", line_width=1, row=2, col=1)
    style_dark_plotly(
        fig,
        height=620,
        xaxis_title=None,
        yaxis_title="Return %",
        margin=dict(t=20, b=34, l=56, r=28),
    )
    fig.update_layout(legend={"orientation": "h", "y": 1.03, "x": 0.0})
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Gross / NAV %", row=3, col=1)
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        config={"displayModeBar": False},
        key="live_performance_stack",
    )


def live_system_read_rows(data: dict[str, Any], latest_nav: pd.DataFrame) -> list[dict[str, object]]:
    snapshot = data["ops_snapshot"]
    ops_rows = pd.DataFrame(snapshot.item_rows)
    fail_count = int(ops_rows["Status"].eq("fail").sum()) if "Status" in ops_rows else 0
    warn_count = int(ops_rows["Status"].eq("warn").sum()) if "Status" in ops_rows else 0
    price_symbols = (
        data["price_history"]["symbol"].nunique()
        if not data["price_history"].empty and "symbol" in data["price_history"]
        else 0
    )
    latest_price_date = (
        human_timestamp(data["price_history"]["date"].max())
        if not data["price_history"].empty and "date" in data["price_history"]
        else "missing"
    )
    latest_as_of = "missing" if latest_nav.empty else human_timestamp(latest_nav.iloc[0].get("as_of"))
    server_sync = data["server_sync"]
    return [
        {
            "Area": "Snapshot",
            "Status": "pass" if latest_as_of != "missing" else "warn",
            "Detail": latest_as_of,
            "Why It Matters": "Most recent unified live account row.",
        },
        {
            "Area": "Ledger",
            "Status": "present" if data["account_ledger"].exists() else "missing",
            "Detail": f"NAV rows={len(data['nav_raw'])}; position rows={len(data['positions'])}",
            "Why It Matters": "Local source powering the dashboard.",
        },
        {
            "Area": "Position Sources",
            "Status": "pass",
            "Detail": f"broker={len(data['broker_positions'])}; manual={len(data['manual_positions'])}; synced={data['manual_sync_rows']}",
            "Why It Matters": "Confirms IBKR and external assets are layered together.",
        },
        {
            "Area": "Market Cache",
            "Status": "present" if not data["price_history"].empty else "missing",
            "Detail": f"rows={len(data['price_history'])}; symbols={price_symbols}; latest={latest_price_date}",
            "Why It Matters": "Feeds HV, benchmark line, and market diagnostics.",
        },
        {
            "Area": "Server Sync",
            "Status": server_sync.get("status", "missing"),
            "Detail": human_timestamp(server_sync.get("synced_at")),
            "Why It Matters": "Shows whether local dashboards received server health state.",
        },
        {
            "Area": "Ops Alerts",
            "Status": snapshot.overall_status,
            "Detail": f"failures={fail_count}; warnings={warn_count}; checked={human_timestamp(snapshot.checked_at)}",
            "Why It Matters": "Summarizes gateway, job, safety, and notification checks.",
        },
    ]


def status_dataframe(rows: list[dict[str, object]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    render_dark_table(frame, empty_message="No status rows available.")


def run_option_quote_refresh() -> tuple[bool, str]:
    script = REPO_ROOT / "scripts" / "refresh_manual_external_holdings.py"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_ROOT), str(REPO_ROOT), existing_pythonpath] if existing_pythonpath else [str(SRC_ROOT), str(REPO_ROOT)]
    )
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=150,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Option quote refresh timed out after 150 seconds."
    except OSError as exc:
        return False, f"Could not start option quote refresh: {exc}"

    output = "\n".join(part for part in (result.stdout, result.stderr) if part.strip()).strip()
    return result.returncode == 0, output or f"Refresh exited with code {result.returncode}."


def live_option_positions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "asset_class" not in frame:
        return pd.DataFrame()
    mask = frame["asset_class"].astype(str).str.lower().str.contains("option", na=False)
    return frame.loc[mask].copy().reset_index(drop=True)


def option_position_label(row: pd.Series | dict[str, Any], index: int) -> str:
    get = row.get if hasattr(row, "get") else dict(row).get
    metadata = read_position_metadata(get("metadata_json"))
    display = metadata.get("display_symbol") or get("symbol") or f"Option {index + 1}"
    underlying = metadata.get("underlying") or get("underlying")
    expiry = metadata.get("expiry") or get("expiry")
    return f"{index + 1}. {display} | {underlying or 'n/a'} | {expiry or 'expiry missing'}"


def selected_option_display(row: pd.Series | dict[str, Any]) -> str:
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    metadata = read_position_metadata(data.get("metadata_json"))
    return str(metadata.get("display_symbol") or data.get("display_symbol") or data.get("symbol") or "option")


def selected_option_expiry(row: pd.Series | dict[str, Any]) -> str:
    legs = extract_portfolio_option_legs(row)
    expiries = sorted({str(leg.expiry) for leg in legs if leg.expiry})
    if len(expiries) == 1:
        return human_date(expiries[0])
    if len(expiries) > 1:
        return " / ".join(human_date(expiry) for expiry in expiries)
    data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    metadata = read_position_metadata(data.get("metadata_json"))
    fallback = metadata.get("expiry") or data.get("expiry")
    return human_date(fallback) if fallback else "missing"


def read_position_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def format_option_risk_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    money_metrics = {
        "Entry Debit",
        "Current P&L",
        "Max Loss",
        "Max Profit",
        "TP 50%",
        "TP 75%",
        "SL 50%",
    }

    def _format(row: pd.Series) -> str:
        metric = str(row.get("Metric") or "")
        value = row.get("Value")
        if metric == "DTE":
            return quantity(value)
        if metric in money_metrics:
            return signed_money(value) if metric != "Entry Debit" else money(value)
        if _float(value) is not None:
            return decimal(value, 2)
        return str(value or "missing")

    out["Value"] = out.apply(_format, axis=1)
    if "Metric" in out:
        out["Metric"] = out["Metric"].map(
            lambda value: currency_header(str(value)) if str(value) in money_metrics else value
        )
    return out


def format_option_greeks_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Strike", "Mark"):
        if column in out:
            out[column] = out[column].map(money)
    if "IV" in out:
        out["IV"] = out["IV"].map(percent)
    for column in ("Delta", "Gamma", "Theta", "Vega"):
        if column in out:
            out[column] = out[column].map(lambda value: decimal(value, 4))
    if "Quantity" in out:
        out["Quantity"] = out["Quantity"].map(quantity)
    out = out.rename(columns={"Strike": currency_header("Strike"), "Mark": currency_header("Mark")})
    return out


def format_position_lab_greeks_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "Leg",
        "Underlying",
        "Expiry",
        "DTE",
        "Type",
        "Strike",
        "Mark",
        "IV",
        "Delta",
        "Gamma $ 1%",
        "Theta / Day",
        "Vega / 1 vol",
        "Model Source",
        "Quality Flag",
    ]
    out = frame[[column for column in columns if column in frame]].copy()
    for column in ("Strike", "Mark"):
        if column in out:
            out[column] = out[column].map(money)
    if "IV" in out:
        out["IV"] = out["IV"].map(percent)
    for column in ("Delta",):
        if column in out:
            out[column] = out[column].map(lambda value: decimal(value, 4))
    for column in ("Gamma $ 1%", "Theta / Day", "Vega / 1 vol"):
        if column in out:
            out[column] = out[column].map(signed_money)
    if "DTE" in out:
        out["DTE"] = out["DTE"].map(quantity)
    out = out.rename(
        columns={
            "Strike": currency_header("Strike"),
            "Mark": currency_header("Mark"),
            "Gamma $ 1%": currency_header("Gamma 1%"),
            "Theta / Day": currency_header("Theta / Day"),
            "Vega / 1 vol": currency_header("Vega / 1 vol"),
        }
    )
    return out


def format_option_book_summary_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    money_columns = {
        "Row Market Value",
        "Row Unrealized P&L",
        "Delta Dollars",
        "Gamma $ 1%",
        "Theta / Day",
        "Vega / 1 vol",
    }
    percent_columns = {"Weighted IV", "Weighted HV 20D"}
    for column in out.columns:
        if column in money_columns:
            out[column] = out[column].map(signed_money)
        elif column in percent_columns:
            out[column] = out[column].map(percent)
        elif column == "IV/HV":
            out[column] = out[column].map(ratio)
        elif column in {"Contracts", "Net Delta Units"}:
            out[column] = out[column].map(quantity)
    out = out.rename(
        columns={
            "Row Market Value": currency_header("Row Market Value"),
            "Row Unrealized P&L": currency_header("Row Unrealized P&L"),
            "Delta Dollars": currency_header("Delta Dollars"),
            "Gamma $ 1%": currency_header("Gamma 1%"),
            "Theta / Day": currency_header("Theta / Day"),
            "Vega / 1 vol": currency_header("Vega / 1 vol"),
        }
    )
    return out


def option_package_register(spreads: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Package",
        "Underlying",
        "Structure",
        "Expiry",
        "DTE",
        "Legs",
        "Contracts",
        "Market Value",
        "Unrealized P&L",
        "Net Delta",
        "Net Theta",
        "Status",
    ]
    if spreads.empty:
        if diagnostics.empty:
            return pd.DataFrame(columns=columns)
        grouped = (
            diagnostics.groupby("Package", dropna=False)
            .agg(
                Underlying=("Underlying", _join_compact),
                Expiry=("Expiry", _join_compact),
                DTE=("DTE", _min_clean_number),
                Legs=("Leg", "count"),
                Contracts=("Quantity", _sum_abs_clean),
                Market_Value=("Mark", _leg_mark_value),
                Unrealized_PnL=("Extrinsic", lambda series: None),
                Net_Delta=("Delta Units", _sum_clean),
                Net_Theta=("Theta / Day", _sum_clean),
                Status=("Quality Flag", _package_status),
            )
            .reset_index()
        )
        grouped["Structure"] = grouped["Legs"].map(lambda value: "single leg" if _float(value) == 1 else "multi-leg package")
        return grouped.rename(
            columns={
                "Market_Value": "Market Value",
                "Unrealized_PnL": "Unrealized P&L",
                "Net_Delta": "Net Delta",
                "Net_Theta": "Net Theta",
            }
        ).reindex(columns=columns)

    out = spreads.copy()
    out["Package"] = out.apply(_package_register_label, axis=1)
    out["Expiry"] = out.get("Expiries", pd.Series("", index=out.index))
    out["Contracts"] = out.get("Net Quantity", pd.Series(pd.NA, index=out.index))
    return out.rename(
        columns={
            "Market Value": "Market Value",
            "Unrealized P&L": "Unrealized P&L",
            "Net Delta": "Net Delta",
            "Net Theta": "Net Theta",
        }
    ).reindex(columns=columns)


def format_option_package_register(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Market Value", "Unrealized P&L", "Net Theta"):
        if column in out:
            formatter = signed_money if column != "Market Value" else money
            out[column] = out[column].map(formatter)
    for column in ("DTE", "Legs", "Contracts"):
        if column in out:
            out[column] = out[column].map(quantity)
    if "Net Delta" in out:
        out["Net Delta"] = out["Net Delta"].map(lambda value: decimal(value, 2))
    return out.rename(
        columns={
            "Market Value": currency_header("Market Value"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
            "Net Theta": currency_header("Net Theta"),
        }
    )


def option_data_quality_table(
    diagnostics: pd.DataFrame,
    option_rows: pd.DataFrame,
    option_legs: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["Check", "Status", "Affected", "Detail", "Next Action"]
    if option_rows.empty:
        return pd.DataFrame(
            [{"Check": "Option inventory", "Status": "pass", "Affected": "0", "Detail": "No option rows in live holdings.", "Next Action": "none"}],
            columns=columns,
        )

    flags = (
        diagnostics.get("Quality Flag", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.lower()
        if not diagnostics.empty
        else pd.Series(dtype=str)
    )
    row_mark_columns = [
        column
        for column in ("market_price", "current_price", "Market Price", "Current Price")
        if column in option_rows
    ]
    if row_mark_columns:
        row_marks = option_rows[row_mark_columns].apply(pd.to_numeric, errors="coerce")
        rows_missing_package_mark = int(row_marks.notna().sum(axis=1).eq(0).sum())
    else:
        rows_missing_package_mark = len(option_rows)
    legs_missing_mark = int(flags.str.contains("mark missing", regex=False).sum())
    rows = []

    def add(check: str, affected: int, detail: str, next_action: str, *, block: bool = False) -> None:
        status = "fail" if block and affected else "warn" if affected else "pass"
        rows.append(
            {
                "Check": check,
                "Status": status,
                "Affected": quantity(affected),
                "Detail": detail,
                "Next Action": next_action if affected else "none",
            }
        )

    parsed_rows = len(option_legs)
    add(
        "Parsed legs",
        0 if parsed_rows else len(option_rows),
        f"{parsed_rows} parsed legs from {len(option_rows)} option rows.",
        "check symbol/metadata shape",
        block=True,
    )
    add(
        "Package marks",
        rows_missing_package_mark,
        "Option rows should have a package/position mark.",
        "refresh Massive/Yahoo option marks",
        block=True,
    )
    add(
        "Leg marks",
        legs_missing_mark,
        "Package mark exists, but some individual legs lack leg-level marks.",
        "fetch leg quotes for better Greeks",
    )
    add("Missing spot", int(flags.str.contains("spot missing", regex=False).sum()), "No underlying spot for model diagnostics.", "refresh underlying prices")
    add("Missing IV", int(flags.str.contains("iv missing", regex=False).sum()), "No implied volatility or solvable IV.", "fetch option quotes/marks")
    add("Missing Greeks", int(flags.str.contains("greeks missing", regex=False).sum()), "No delta/gamma/theta/vega.", "compute via model once mark, spot, IV exist")
    add("Expired", int(flags.str.contains("expired", regex=False).sum()), "Expired contracts still in the book.", "archive or reconcile position")
    add("Below intrinsic", int(flags.str.contains("below intrinsic", regex=False).sum()), "Mark is below intrinsic value.", "inspect quote quality")
    return pd.DataFrame(rows, columns=columns)


def format_option_data_quality_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    return out.rename(columns={"Next Action": "Action"})


def option_underlying_book_audit(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Symbol",
        "Market Value",
        "Option Market Value",
        "Option Cost Basis",
        "Unrealized P&L",
        "Gross Option Contract Shares",
        "Net Option Delta",
        "Delta Dollars",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    option_value = pd.to_numeric(out.get("Option Market Value", 0.0), errors="coerce").fillna(0.0).abs()
    gross_shares = pd.to_numeric(out.get("Gross Option Contract Shares", 0.0), errors="coerce").fillna(0.0).abs()
    out = out.loc[option_value.gt(0) | gross_shares.gt(0)]
    return out.reindex(columns=columns)


def format_option_underlying_audit_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    display = pd.DataFrame(
        {
            "Underlying": out.get("Symbol"),
            "Mkt Value": out.get("Market Value"),
            "Opt Value": out.get("Option Market Value"),
            "Our Cost": out.get("Option Cost Basis"),
            "P&L": out.get("Unrealized P&L"),
            "Gross Shares": out.get("Gross Option Contract Shares"),
            "Delta Shares": out.get("Net Option Delta"),
            "Delta $": out.get("Delta Dollars"),
        }
    )
    for column in ("Mkt Value", "Opt Value", "Our Cost", "Delta $"):
        display[column] = display[column].map(money)
    display["P&L"] = display["P&L"].map(signed_money)
    display["Gross Shares"] = display["Gross Shares"].map(quantity)
    display["Delta Shares"] = display["Delta Shares"].map(lambda value: decimal(value, 2))
    display = display.rename(
        columns={
            "Mkt Value": currency_header("Mkt Value"),
            "Opt Value": currency_header("Opt Value"),
            "Our Cost": currency_header("Our Cost"),
            "P&L": currency_header("P&L"),
            "Delta $": currency_header("Delta Exposure"),
        }
    )
    return display


def format_option_leg_audit_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    display_columns = [
        "Spread Group",
        "Symbol",
        "Underlying",
        "Expiry",
        "DTE",
        "Type",
        "Strike",
        "Quantity",
        "Market Price",
        "Market Value",
        "Unrealized P&L",
        "Delta",
        "Theta",
        "Vega",
        "HV 20D",
    ]
    out = out[[column for column in display_columns if column in out]]
    for column in ("Strike", "Market Price", "Market Value"):
        if column in out:
            out[column] = out[column].map(money)
    if "Unrealized P&L" in out:
        out["Unrealized P&L"] = out["Unrealized P&L"].map(signed_money)
    for column in ("Delta", "Theta", "Vega"):
        if column in out:
            out[column] = out[column].map(lambda value: decimal(value, 4))
    if "HV 20D" in out:
        out["HV 20D"] = out["HV 20D"].map(percent)
    for column in ("DTE", "Quantity"):
        if column in out:
            out[column] = out[column].map(quantity)
    return out.rename(
        columns={
            "Strike": currency_header("Strike"),
            "Market Price": currency_header("Market Price"),
            "Market Value": currency_header("Market Value"),
            "Unrealized P&L": currency_header("Unrealized P&L"),
        }
    )


def _join_compact(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna().unique().tolist() if str(value).strip()]
    return ", ".join(values[:4]) + ("" if len(values) <= 4 else f" +{len(values) - 4}")


def _package_register_label(row: pd.Series) -> str:
    underlying = str(row.get("Underlying") or "").strip()
    structure = str(row.get("Structure") or "").strip()
    expiries = str(row.get("Expiries") or "").strip()
    dte = quantity(row.get("DTE")) if row.get("DTE") is not None else "DTE n/a"
    parts = [part for part in (underlying, structure, expiries or dte) if part and part != "missing"]
    return " | ".join(parts) or str(row.get("Spread ID") or "option package")


def _sum_clean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).sum())


def _sum_abs_clean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).abs().sum())


def _min_clean_number(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return None if clean.empty else float(clean.min())


def _package_status(series: pd.Series) -> str:
    flags = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
    if not flags:
        return "missing"
    if all(value == "ok" for value in flags):
        return "ok"
    if any("mark missing" in value or "expired" in value for value in flags):
        return "review"
    return "watch"


def _leg_mark_value(series: pd.Series) -> float | None:
    # Used only by the diagnostics fallback; leg marks lack multiplier context here.
    clean = pd.to_numeric(series, errors="coerce")
    return None if clean.dropna().empty else float(clean.sum())


def format_option_diagnostics_display(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column in ("Strike", "Spot", "Entry", "Mark", "Intrinsic", "Extrinsic"):
        if column in out:
            out[column] = out[column].map(money)
    for column in ("IV", "HV 20D"):
        if column in out:
            out[column] = out[column].map(percent)
    if "IV/HV" in out:
        out["IV/HV"] = out["IV/HV"].map(ratio)
    if "Moneyness" in out:
        out["Moneyness"] = out["Moneyness"].map(lambda value: decimal(value, 3))
    for column in ("Delta", "Gamma $ 1%", "Theta / Day", "Vega / 1 vol", "Delta Dollars"):
        if column in out and column == "Delta":
            out[column] = out[column].map(lambda value: decimal(value, 4))
        elif column in out:
            out[column] = out[column].map(signed_money)
    for column in ("Quantity", "DTE", "Delta Units"):
        if column in out:
            out[column] = out[column].map(quantity)
    display_columns = [
        "Package",
        "Leg",
        "Underlying",
        "Expiry",
        "DTE",
        "Moneyness",
        "Mark",
        "Intrinsic",
        "Extrinsic",
        "IV",
        "HV 20D",
        "IV/HV",
        "Delta",
        "Delta Dollars",
        "Gamma $ 1%",
        "Theta / Day",
        "Vega / 1 vol",
        "Model Source",
        "Quality Flag",
    ]
    out = out[[column for column in display_columns if column in out]]
    return out.rename(
        columns={
            "Mark": currency_header("Mark"),
            "Intrinsic": currency_header("Intrinsic"),
            "Extrinsic": currency_header("Extrinsic"),
            "Delta Dollars": currency_header("Delta Dollars"),
            "Gamma $ 1%": currency_header("Gamma 1%"),
            "Theta / Day": currency_header("Theta / Day"),
            "Vega / 1 vol": currency_header("Vega / 1 vol"),
        }
    )


def render_option_payoff_curve(row: pd.Series | dict[str, Any]) -> None:
    curve = option_payoff_curve(row)
    if curve.empty:
        st.info("No payoff curve can be built for this option row yet.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["Underlying Price"],
            y=curve["Expiry P&L"],
            mode="lines",
            name="Expiry P&L",
            line=dict(color="#5eead4", width=3),
        )
    )
    fig.add_hline(y=0, line=dict(color="rgba(148, 163, 184, 0.45)", width=1, dash="dot"))
    style_dark_plotly(
        fig,
        height=360,
        margin=dict(t=16, b=32, l=44, r=16),
        hovermode="x unified",
    )
    fig.update_layout(xaxis_title="Underlying Price", yaxis_title="P&L at Expiry")
    st.plotly_chart(fig, use_container_width=True, theme=None)


def render_option_payoff_surface(row: pd.Series | dict[str, Any]) -> None:
    surface = option_payoff_surface(row)
    if surface.empty:
        st.info("No payoff surface can be built for this option row yet.")
        return
    pivot = surface.pivot_table(
        index="Days To Expiry",
        columns="Underlying Price",
        values="Illustrative P&L",
        aggfunc="mean",
    ).sort_index()
    fig = go.Figure(
        data=[
            go.Surface(
                x=pivot.columns.astype(float),
                y=pivot.index.astype(float),
                z=pivot.values,
                colorscale=[
                    [0.0, "#fb7185"],
                    [0.5, "#111827"],
                    [1.0, "#5eead4"],
                ],
                colorbar=dict(title="P&L"),
            )
        ]
    )
    style_dark_plotly(
        fig,
        height=430,
        margin=dict(t=18, b=18, l=12, r=12),
        hovermode=None,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="Underlying",
            yaxis_title="Days To Expiry",
            zaxis_title="Illustrative P&L",
            xaxis=dict(backgroundcolor="#07101a", gridcolor="rgba(148,163,184,0.18)"),
            yaxis=dict(backgroundcolor="#07101a", gridcolor="rgba(148,163,184,0.18)"),
            zaxis=dict(backgroundcolor="#07101a", gridcolor="rgba(148,163,184,0.18)"),
        )
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


data = load_live_portfolio_data()
latest_nav = data["latest_nav"]
nav_raw = data["nav_raw"]
nav_history = data["nav_history"]
all_positions = data["positions"]
position_history = data["position_history"]
events = data["events"]
hv = data["hv"]
all_holdings = enriched_live_holdings(all_positions, hv)
REPORTING_CURRENCY = latest_nav_currency(latest_nav, data["unified_write"].get("currency"))

page_header(
    title="Live Portfolio",
    title_zh="实盘组合",
    subtitle="Real-money monitoring room for account health, holdings, performance, exposure, and reconciliation.",
    subtitle_zh="真实资金账户的健康、持仓、表现、敞口与对账监控室。",
    language=OPS_LANG,
)
account_scope = account_wrapper_selector()
if account_scope == US_ACCOUNT_SCOPE:
    st.caption("IBKR live account plus external/manual holdings that are not routed through QMT.")
else:
    st.caption("华源证券/QMT account wrapper for CN equities, CN options, and CN futures. Empty states are expected until the connector feeds positions.")

positions = positions_for_account_scope(all_positions, data["all_live_positions"], account_scope)
holdings = holdings_for_account_scope(all_holdings, account_scope)
if account_scope == CN_ACCOUNT_SCOPE and holdings.empty and not positions.empty:
    holdings = enriched_live_holdings(positions, hv)
filtered_holdings = holdings

latest_nav_for_view = latest_nav if account_scope == US_ACCOUNT_SCOPE else pd.DataFrame()
nav_raw_for_view = nav_raw if account_scope == US_ACCOUNT_SCOPE else pd.DataFrame(columns=nav_raw.columns)
nav_history_for_view = nav_history if account_scope == US_ACCOUNT_SCOPE else pd.DataFrame(columns=nav_history.columns)

nav = (
    latest_nav_value(latest_nav, "net_liquidation")
    if account_scope == US_ACCOUNT_SCOPE
    else nav_from_holdings(holdings)
)
cash = (
    latest_nav_value(latest_nav, "cash")
    if account_scope == US_ACCOUNT_SCOPE
    else cash_from_holdings(holdings)
)
daily_pnl = latest_nav_value(latest_nav, "daily_pnl") if account_scope == US_ACCOUNT_SCOPE else None
cash_like_value = cash_like_holdings_value(holdings)
total_cash_value = cash_with_equivalents(cash, cash_like_value)
allocation_cash = cash_not_already_in_holdings(holdings, cash)
performance = account_performance_summary(
    nav_raw_for_view,
    positions,
    current_nav=nav,
    current_cash=cash,
    current_daily_pnl=daily_pnl,
)
if account_scope == CN_ACCOUNT_SCOPE:
    performance["unrealized_pnl"] = pnl_from_holdings(holdings)
    gross_value = nav_from_holdings(holdings)
    performance["gross_exposure_pct"] = None if not nav or not gross_value else gross_value / abs(float(nav))
spreads = recognize_option_spreads(positions, hv)
option_legs = option_leg_report(positions, hv)
underlying_exposure = underlying_exposure_report(positions, hv)
option_adjusted_underlying = clean_underlying_rollup(option_adjusted_underlying_frame(underlying_exposure, nav=nav))
market_risk = compute_market_risk_decomposition(
    option_adjusted_underlying,
    data["price_history"],
    config=MarketRiskConfig(benchmark="QQQ"),
)
option_positions = live_option_positions(positions)
option_book = option_book_summary(positions, hv)
option_diagnostics = option_position_diagnostics(positions, hv)
sleeve_mix = asset_sleeve_mix(holdings, cash=allocation_cash)
risk_positions = position_risk_frame(holdings, nav=nav, cash=allocation_cash)
sector_mix = sector_exposure_frame(holdings, cash=allocation_cash)
concentration_metrics = concentration_diagnostics_frame(risk_positions)
currency_mix = currency_exposure_frame(
    holdings,
    cash=allocation_cash,
    cash_currency=cash_currency_for_holdings(holdings),
)

top = st.columns(8)
top[0].metric(f"{T('nav')} ({REPORTING_CURRENCY})", money(nav))
top[1].metric(
    f"{T('total_cash', 'Total Cash')} ({REPORTING_CURRENCY})",
    money(total_cash_value),
    help="Broker cash plus manual external cash plus cash-equivalent holdings such as XEON, converted into the dashboard reporting currency.",
)
top[2].metric(f"{T('daily_pnl')} ({REPORTING_CURRENCY})", signed_money(daily_pnl))
top[3].metric(f"{T('unrealized_pnl')} ({REPORTING_CURRENCY})", signed_money(performance.get("unrealized_pnl")))
top[4].metric(T("positions"), str(len(positions)))
top[5].metric(
    T("gross_nav"),
    percent(performance.get("gross_exposure_pct")),
    help="Gross exposure divided by NAV. This is an exposure/leverage read, not profit divided by NAV.",
)
portfolio_beta = _float(market_risk.get("portfolio_beta")) if market_risk.get("status") == "live" else None
top[6].metric(
    "PORTFOLIO BETA",
    "missing" if portfolio_beta is None else f"{portfolio_beta:.2f}",
    help=(
        "Current-weight portfolio beta relative to QQQ. Missing asset history is reported as uncovered "
        "rather than assigned beta zero; see Exposure for coverage and decomposition."
    ),
)
snapshot_time = (
    human_timestamp(latest_nav.iloc[0].get("as_of"))
    if account_scope == US_ACCOUNT_SCOPE and not latest_nav.empty
    else (
        human_timestamp(holdings["As Of"].dropna().max())
        if not holdings.empty and "As Of" in holdings and not holdings["As Of"].dropna().empty
        else T("missing")
    )
)
top[7].metric(T("snapshot"), snapshot_time)

if account_scope == US_ACCOUNT_SCOPE and not data["manual_positions"].empty:
    manual_currencies = ", ".join(
        sorted(
            data["manual_positions"]["currency"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )
    )
    st.info(
        f"{len(data['manual_positions'])} external/manual holding rows are layered into the holdings view "
        f"({manual_currencies}). Synced from {display_path(data['manual_external_json'])}. "
        "Rows marked manual_cost are cost-basis placeholders until a pricing/FX refresh is approved."
    )

overview_tab, holdings_tab, exposure_tab, spreads_tab, reconciliation_tab = st.tabs(
    ops_tabs(OPS_LANG, "live_tabs")
)

with overview_tab:
    st.subheader("Portfolio Performance Stack")
    st.caption("Portfolio return, benchmark when available, drawdown, and current gross exposure/NAV in one scan.")
    render_live_performance_stack(nav_history_for_view, data["price_history"], performance)

    st.subheader("Weekly Performance")
    weekly = weekly_return_table(nav_history_for_view)
    if weekly.empty:
        st.info("No weekly performance table is available yet.")
    else:
        render_dark_table(weekly, max_height_px=320)

    mix_left, mix_right = st.columns(2)
    with mix_left:
        st.subheader("Asset Sleeve Mix")
        if sleeve_mix.empty:
            st.info("No live allocation sleeves are available yet.")
        else:
            render_dark_pie_chart(
                sleeve_mix,
                names="Sleeve",
                values="Exposure Value",
                empty_message="No live allocation sleeves are available yet.",
                height=300,
            )
            sleeve_table = sleeve_mix.copy()
            sleeve_table["Exposure Value"] = sleeve_table["Exposure Value"].map(money)
            sleeve_table["Weight"] = sleeve_table["Weight"].map(percent)
            sleeve_table = sleeve_table.rename(columns={"Exposure Value": currency_header("Exposure Value")})
            render_dark_table(sleeve_table, max_height_px=220)
    with mix_right:
        st.subheader("Daily P&L")
        if nav_history_for_view.empty:
            st.info("No live daily P&L history has been recorded yet.")
        else:
            render_dark_bar_chart(
                nav_history_for_view.set_index("date")[["daily_pnl"]],
                yaxis_title=currency_header("Daily P&L"),
                height=360,
                positive_color="#22C55E",
                negative_color="#EF4444",
            )

    if account_scope == CN_ACCOUNT_SCOPE:
        st.subheader("QMT Live Monitor")
        render_qmt_connector_panel(data["settings"], data["ops_snapshot"], compact=True)

with holdings_tab:
    planned_holdings = add_holding_plan_columns(
        filtered_holdings,
        data["atr_price_history"],
        styles=load_holding_styles(DEFAULT_HOLDING_PLAN_PATH),
    )
    st.subheader("Current Holdings")
    if planned_holdings.empty:
        st.info("No live holdings rows match the current filters.")
    elif not render_holdings_aggrid(planned_holdings):
        st.warning("`streamlit-aggrid` is not installed, so current holdings are using the static dark-table fallback.")
        render_dark_table(format_holdings_display(planned_holdings), max_height_px=520)
    if hv.empty:
        st.info("HV columns are waiting for a price-history cache such as runtime/market/price_history.parquet.")

    winners, losers = winners_losers_tables(risk_positions)
    win_left, lose_right = st.columns(2)
    with win_left:
        st.subheader("Top Winners")
        if winners.empty:
            st.info("No winning positions are available.")
        else:
            render_dark_table(winners, max_height_px=260)
    with lose_right:
        st.subheader("Top Losers")
        if losers.empty:
            st.info("No losing positions are available.")
        else:
            render_dark_table(losers, max_height_px=260)

    if account_scope == CN_ACCOUNT_SCOPE:
        st.subheader("QMT/CN Holdings Slice")
        render_dark_table(
            qmt_position_slice(data["all_live_positions"], environment="live"),
            empty_message="No QMT live holdings are available yet.",
            max_height_px=360,
        )

with spreads_tab:
    st.caption("Options command room for spread recognition, payoff shape, Greeks, and TP/SL scaffolding.")
    refresh_notice = st.session_state.pop("live_option_quote_refresh_notice", None)
    if refresh_notice:
        st.success("Option quotes refreshed. The account ledger has been reloaded from the latest marks.")
        if st.toggle("Show refresh output", value=False, key="live_option_quote_refresh_output"):
            st.code(str(refresh_notice)[-4000:])

    refresh_left, refresh_right = st.columns([0.18, 0.82])
    with refresh_left:
        refresh_clicked = st.button(
            "Refresh Option Quotes",
            key="live_refresh_option_quotes",
            type="primary",
            use_container_width=True,
        )
    with refresh_right:
        st.caption(
            "Pulls manual external option marks from Massive first, then Yahoo fallback, "
            "and stores marks, source metadata, FX, and underlying spot in the account ledger."
        )
    if refresh_clicked:
        with st.spinner("Refreshing option marks from Massive/Yahoo..."):
            ok, refresh_output = run_option_quote_refresh()
        if ok:
            st.session_state["live_option_quote_refresh_notice"] = refresh_output
            st.rerun()
        else:
            st.error("Option quote refresh failed.")
            st.code(refresh_output[-4000:])

    option_stats = st.columns(6)
    option_market_value = pd.to_numeric(option_positions.get("market_value", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    option_unrealized = pd.to_numeric(option_positions.get("unrealized_pnl", pd.Series(dtype=float)), errors="coerce").sum(min_count=1)
    option_book_row = option_book.iloc[0].to_dict() if not option_book.empty else {}
    option_stats[0].metric("Option Rows", str(len(option_positions)))
    option_stats[1].metric("Recognized Spreads", str(len(spreads)))
    option_stats[2].metric(f"Option Market Value ({REPORTING_CURRENCY})", money(option_market_value))
    option_stats[3].metric(f"Option Unrealized P&L ({REPORTING_CURRENCY})", signed_money(option_unrealized))
    option_stats[4].metric(f"Net Delta ({REPORTING_CURRENCY})", signed_money(option_book_row.get("Delta Dollars")))
    option_stats[5].metric(f"Theta / Day ({REPORTING_CURRENCY})", signed_money(option_book_row.get("Theta / Day")))

    audit_tab, position_lab_tab = st.tabs(ops_tabs(OPS_LANG, "option_tabs")[:2])

    with audit_tab:
        package_register = option_package_register(spreads, option_diagnostics)
        option_underlying_audit = option_underlying_book_audit(option_adjusted_underlying)

        st.subheader("Package Register")
        st.caption("One row per recognized option package. This is the inventory list before payoff/risk analysis.")
        if package_register.empty:
            st.info("No option packages are recognized in current holdings.")
        else:
            render_dark_table(format_option_package_register(package_register), max_height_px=360)

        st.subheader("Underlying Rollup")
        st.caption(
            "Only option-linked underlyings are shown here; cash/equity-only rows stay in Exposure. "
            "Our Cost is option market value minus unrealized P&L. Delta Shares is share-equivalent delta after "
            "contract multipliers and netting; Delta Exposure is Delta Shares × underlying spot."
        )
        if option_underlying_audit.empty:
            st.info("No option-linked underlying exposure rows are available.")
        else:
            render_dark_table(format_option_underlying_audit_display(option_underlying_audit), max_height_px=420)

        st.subheader("Position Model Audit")
        if option_diagnostics.empty:
            st.info("No option diagnostics are available.")
        else:
            render_dark_table(
                format_option_diagnostics_display(option_diagnostics),
                max_height_px=520,
                min_width_px=2200,
                nowrap=True,
            )

    with position_lab_tab:
        st.subheader("Position Lab")
        st.caption(
            "One selector controls the payoff chart, 3D surface, Greeks, and checkpoints for the same option package."
        )
        if option_positions.empty:
            st.info("No option positions are available for payoff or Greek diagnostics.")
        else:
            labels = [option_position_label(row, index) for index, row in option_positions.iterrows()]
            selected_label = st.selectbox(
                T("option_package_leg", "Option package / leg"),
                labels,
                index=0,
                key="live_option_position_lab_selector",
            )
            selected_index = labels.index(selected_label)
            selected_row = option_positions.iloc[selected_index]
            selected_diagnostics = option_position_diagnostics(pd.DataFrame([selected_row]), hv)

            selected_cards = st.columns(5)
            selected_cards[0].metric("Selected", selected_option_display(selected_row))
            selected_cards[1].metric(currency_header("Mark"), money(selected_row.get("market_price")))
            selected_cards[2].metric(currency_header("Market Value"), money(selected_row.get("market_value")))
            selected_cards[3].metric(currency_header("Unrealized P&L"), signed_money(selected_row.get("unrealized_pnl")))
            selected_cards[4].metric("Expiry", selected_option_expiry(selected_row))

            st.subheader("Payoff Shape")
            chart_left, chart_right = st.columns([1.05, 1.15])
            with chart_left:
                st.caption("Expiry payoff at different underlying prices.")
                render_option_payoff_curve(selected_row)
            with chart_right:
                st.caption(
                    "Illustrative price/time surface. A true mark-to-market surface needs a fitted volatility surface "
                    "from the option chain across strikes and expiries."
                )
                render_option_payoff_surface(selected_row)

            st.subheader("Greeks & Risk")
            greek_left, risk_right = st.columns([1.05, 0.95])
            with greek_left:
                st.caption("Leg-level Greeks from Massive when available; otherwise solved IV plus Black-Scholes fallback where spot/mark are present.")
                greek_columns = [
                    column
                    for column in ("IV", "Delta", "Gamma $ 1%", "Theta / Day", "Vega / 1 vol")
                    if column in selected_diagnostics.columns
                ]
                greek_numeric = (
                    selected_diagnostics[greek_columns].apply(pd.to_numeric, errors="coerce")
                    if not selected_diagnostics.empty and greek_columns
                    else pd.DataFrame()
                )
                if selected_diagnostics.empty:
                    st.info("No option legs are available for this row.")
                elif greek_numeric.empty or greek_numeric.notna().sum().sum() == 0:
                    st.info(
                        "Greeks still need an underlying spot, option mark, and expiry. "
                        "Run the manual holdings refresh to populate spot/mark metadata, then the model fallback can solve IV/Greeks."
                    )
                    render_dark_table(format_position_lab_greeks_display(selected_diagnostics), max_height_px=320)
                else:
                    render_dark_table(format_position_lab_greeks_display(selected_diagnostics), max_height_px=360)
            with risk_right:
                st.caption("Decision scaffolding for TP/SL, defined risk, and quote-quality checks.")
                risk_summary = option_risk_summary(selected_row)
                render_dark_table(format_option_risk_display(risk_summary), max_height_px=360)

with exposure_tab:
    st.caption(
        "Risk lab view: marked allocation, option-adjusted underlying exposure, concentration, "
        "and currency mix."
    )
    if account_scope == CN_ACCOUNT_SCOPE:
        st.subheader("QMT/CN Exposure Slice")
        render_dark_table(
            qmt_exposure_by_asset(data["all_live_positions"]),
            empty_message="No QMT exposure rows are available yet.",
            max_height_px=260,
        )

    non_cash_sectors = sector_mix.loc[sector_mix["Sector"].astype(str) != "Cash"] if not sector_mix.empty else pd.DataFrame()
    largest_sector_row = (
        non_cash_sectors.sort_values("Weight", ascending=False).head(1)
        if not non_cash_sectors.empty
        else pd.DataFrame()
    )
    cash_sector = sector_mix.loc[sector_mix["Sector"].astype(str) == "Cash"] if not sector_mix.empty else pd.DataFrame()
    largest_underlying_row = (
        option_adjusted_underlying.sort_values("Weight", ascending=False).head(1)
        if not option_adjusted_underlying.empty
        else pd.DataFrame()
    )
    gross_contract_shares = None
    if not option_adjusted_underlying.empty and "Gross Option Contract Shares" in option_adjusted_underlying:
        gross_contract_shares = pd.to_numeric(
            option_adjusted_underlying["Gross Option Contract Shares"],
            errors="coerce",
        ).sum(min_count=1)
        gross_contract_shares = None if pd.isna(gross_contract_shares) else float(gross_contract_shares)

    st.subheader("Portfolio Risk & Exposure")
    st.caption("QQQ beta/risk decomposition alongside the key current-exposure diagnostics.")
    exposure_stats = st.columns(10)
    exposure_stats[0].metric(
        "Portfolio Beta",
        "missing" if market_risk.get("status") != "live" else decimal(market_risk.get("portfolio_beta"), 2),
    )
    exposure_stats[1].metric("Total Vol", percent(market_risk.get("total_volatility")))
    exposure_stats[2].metric(
        "Systematic Risk",
        percent(market_risk.get("systematic_risk_share")),
        help="Share of modeled portfolio variance explained by QQQ.",
    )
    exposure_stats[3].metric(
        "Idiosyncratic Risk",
        percent(market_risk.get("idiosyncratic_risk_share")),
        help="Share of modeled portfolio variance left in the regression residual.",
    )
    exposure_stats[4].metric(
        "History Coverage",
        percent(market_risk.get("covered_weight")),
        help="Share of current non-cash economic exposure with at least 60 overlapping daily returns.",
    )
    exposure_stats[5].metric(
        "Largest Sector",
        "missing" if largest_sector_row.empty else str(largest_sector_row.iloc[0]["Sector"]),
        help="Weight: " + ("missing" if largest_sector_row.empty else percent(largest_sector_row.iloc[0].get("Weight"))),
    )
    exposure_stats[6].metric(
        "Cash-Like Weight",
        "missing" if cash_sector.empty else percent(cash_sector.iloc[0].get("Weight")),
        help="Cash plus cash-equivalent holdings such as XEON, expressed as a share of marked exposure.",
    )
    exposure_stats[7].metric(
        "Largest Underlying",
        "missing" if largest_underlying_row.empty else str(largest_underlying_row.iloc[0]["Symbol"]),
        help="Weight: " + ("missing" if largest_underlying_row.empty else percent(largest_underlying_row.iloc[0].get("Weight"))),
    )
    exposure_stats[8].metric(
        "Top 5 Marked Weight",
        percent(concentration_value(concentration_metrics, "Top 5 weight")),
    )
    exposure_stats[9].metric("Gross Option Shares", quantity(gross_contract_shares))

    risk_split = pd.DataFrame(
        {
            "Risk": ["Systematic", "Idiosyncratic"],
            "Variance Share": [
                market_risk.get("systematic_risk_share"),
                market_risk.get("idiosyncratic_risk_share"),
            ],
        }
    ).dropna(subset=["Variance Share"])
    position_beta = market_risk.get("positions", pd.DataFrame()).copy()
    if not risk_split.empty or not position_beta.empty:
        risk_chart_col, risk_table_col = st.columns([0.72, 1.28])
        with risk_chart_col:
            if not risk_split.empty:
                render_dark_pie_chart(
                    risk_split,
                    names="Risk",
                    values="Variance Share",
                    empty_message="No market-risk decomposition is available.",
                    height=310,
                )
        with risk_table_col:
            if not position_beta.empty:
                for column in ("Weight", "Annualized Volatility"):
                    if column in position_beta:
                        position_beta[column] = position_beta[column].map(percent)
                for column in ("Beta", "Beta Contribution", "Correlation"):
                    if column in position_beta:
                        position_beta[column] = position_beta[column].map(lambda value: decimal(value, 3))
                for column in ("History Start", "History End"):
                    if column in position_beta:
                        position_beta[column] = position_beta[column].map(human_timestamp)
                if "Observations" in position_beta:
                    position_beta["Observations"] = position_beta["Observations"].map(quantity)
                render_dark_table(position_beta, max_height_px=360, min_width_px=1250, nowrap=True)

    risk_modules = pd.DataFrame(
        [
            {
                "Layer": "Marked allocation",
                "Status": "live",
                "Purpose": "Sector, sleeve, currency, and market-value concentration.",
                "Main Caveat": "Options appear by their marked premium value.",
            },
            {
                "Layer": "Underlying exposure",
                "Status": "live",
                "Purpose": "Roll options into their underlying ticker and show contract-share leverage.",
                "Main Caveat": "Delta dollars need live Greeks/spot to become fully economic.",
            },
            {
                "Layer": "VaR / CVaR",
                "Status": "planned",
                "Purpose": "Estimate tail loss under historical or simulated shocks.",
                "Main Caveat": "Options need scenario repricing, not just premium marks.",
            },
            {
                "Layer": "MVO / allocation lab",
                "Status": "planned",
                "Purpose": "Suggest risk-aware target weights under your constraints.",
                "Main Caveat": "Should consume centralized portfolio policy limits.",
            },
        ]
    )

    sector_left, sector_right = st.columns([1.05, 1.0])
    with sector_left:
        st.subheader("Marked Sector Exposure")
        st.caption("Uses current market value. Option rows inherit the sector of their underlying ticker.")
        if sector_mix.empty:
            st.info("No sector exposure is available.")
        else:
            render_dark_pie_chart(
                sector_mix,
                names="Sector",
                values="Exposure Value",
                empty_message="No sector exposure is available.",
                height=340,
            )
    with sector_right:
        st.subheader("Concentration Diagnostics")
        st.caption("Marked-position concentration, including cash and cash equivalents.")
        render_dark_table(
            format_concentration_table(concentration_metrics),
            empty_message="No concentration diagnostics are available.",
            max_height_px=360,
        )
    if not sector_mix.empty:
        st.subheader("Sector Detail")
        render_dark_table(format_exposure_table(sector_mix), max_height_px=300)

    st.subheader("Option-Adjusted Underlying Exposure")
    st.caption(
        "Tencent-style aliases are canonicalized, so 0700.HK option exposure and TCEHY/Tencent stock exposure "
        "roll into the same underlying. Contract-share columns show controlled shares; delta dollars appear when "
        "the option data source supplies usable Greeks and spot."
    )
    if option_adjusted_underlying.empty:
        st.info("No underlying exposure rows are available.")
    else:
        underlying_left, underlying_right = st.columns([1.15, 0.85])
        with underlying_left:
            render_horizontal_exposure_bar(
                option_adjusted_underlying.rename(columns={"Economic Exposure": "Exposure Value"}),
                label_column="Symbol",
                height=340,
                key="live_option_adjusted_underlying_bar",
            )
        with underlying_right:
            share_frame = option_adjusted_underlying.reindex(
                columns=["Symbol", "Gross Option Contract Shares"]
            ).copy()
            share_frame["Exposure Value"] = pd.to_numeric(
                share_frame["Gross Option Contract Shares"],
                errors="coerce",
            ).abs()
            render_horizontal_exposure_bar(
                share_frame,
                label_column="Symbol",
                height=340,
                key="live_option_contract_share_bar",
            )
        render_dark_table(format_option_adjusted_underlying_table(option_adjusted_underlying), max_height_px=360)

    st.subheader("Sleeve & Currency")
    sleeve_inner, currency_inner = st.columns(2)
    with sleeve_inner:
        if sleeve_mix.empty:
            st.info("No sleeve exposure is available.")
        else:
            render_dark_pie_chart(
                sleeve_mix,
                names="Sleeve",
                values="Exposure Value",
                empty_message="No sleeve exposure is available.",
                height=280,
            )
    with currency_inner:
        if currency_mix.empty:
            st.info("No currency exposure is available.")
        else:
            render_dark_pie_chart(
                currency_mix,
                names="Currency",
                values="Exposure Value",
                empty_message="No currency exposure is available.",
                height=280,
            )
    if not currency_mix.empty:
        render_dark_table(format_exposure_table(currency_mix), max_height_px=210)

with reconciliation_tab:
    st.subheader("Account Ledger")
    status_dataframe(
        [
            {"Item": "Path", "Value": display_path(data["account_ledger"])},
            {"Item": "Account wrapper", "Value": account_scope},
            {"Item": "NAV rows", "Value": str(len(nav_raw_for_view))},
            {"Item": "Position rows", "Value": str(len(positions))},
            {"Item": "IBKR/broker position rows", "Value": str(len(data["broker_positions"]))},
            {"Item": "External/manual position rows", "Value": str(len(data["manual_positions"]))},
            {"Item": "External/manual JSON", "Value": display_path(data["manual_external_json"])},
            {"Item": "External/manual rows synced", "Value": str(data["manual_sync_rows"])},
            {"Item": "Unified profile", "Value": UNIFIED_LIVE_PROFILE},
            {"Item": f"Unified NAV written ({REPORTING_CURRENCY})", "Value": money(data["unified_write"].get("net_liquidation"))},
            {"Item": f"Unified manual value ({REPORTING_CURRENCY})", "Value": money(data["unified_write"].get("manual_usd_value"))},
            {"Item": f"Manual rows excluded from {REPORTING_CURRENCY} NAV", "Value": str(data["unified_write"].get("excluded_manual_rows", 0))},
            {"Item": "Position history rows", "Value": str(len(position_history))},
            {"Item": "Latest snapshot", "Value": "missing" if latest_nav_for_view.empty else human_timestamp(latest_nav_for_view.iloc[0].get("as_of"))},
            {"Item": "Server runtime sync", "Value": human_timestamp(data["server_sync"].get("synced_at"))},
            {"Item": "Price history rows", "Value": str(len(data["price_history"]))},
            {"Item": "HV symbols", "Value": str(len(hv))},
        ]
    )

    if account_scope == CN_ACCOUNT_SCOPE:
        st.subheader("QMT Live Account Reconciliation")
        render_qmt_account_panel(data["ops_snapshot"], data["all_live_positions"], environment="live")

        st.subheader("QMT Connector Contracts")
        render_dark_table(qmt_connector_contract_frame(data["settings"]), max_height_px=260)

    st.subheader("Risk Lab Structure")
    st.caption(
        "Implementation map for the exposure and portfolio-risk modules. This is an operational checklist, "
        "so it lives here with the reconciliation and data-readiness checks."
    )
    render_dark_table(risk_modules, max_height_px=300)

    st.subheader("Ops Checks")
    ops_rows = pd.DataFrame(data["ops_snapshot"].item_rows)
    if ops_rows.empty:
        st.info("No ops status rows are available.")
    else:
        live_related = ops_rows[
            ops_rows["Category"].isin(["Gateway", "Broker Heartbeat", "Accounts", "Jobs", "Safety"])
        ]
        status_dataframe(live_related if not live_related.empty else ops_rows)

    if account_scope == CN_ACCOUNT_SCOPE:
        st.subheader("QMT Audit Trail")
        render_qmt_audit_panel(data["settings"], limit=10)

    st.subheader("Latest Live Events")
    if events.empty:
        st.info("No live account events have been recorded yet.")
    else:
        left, right = st.columns([1.3, 1])
        with left:
            render_dark_table(account_trade_events_display(events), max_height_px=420)
        with right:
            summary = account_trade_event_summary(events)
            render_dark_table(summary, empty_message="No live event summary is available.")

if account_scope == US_ACCOUNT_SCOPE:
    st.divider()
    with st.expander("Edit Manual External Holdings JSON", expanded=False):
        st.caption(
            f"Source of truth: {display_path(data['manual_external_json'])}. "
            "Saving validates every row, creates a .bak copy, synchronizes the account ledger, "
            "and immediately recalculates the page."
        )
        st.warning(
            "Changes affect the live monitoring view. Keep position_id values unique and review "
            "quantity, multiplier, currency, prices, and FX rates before saving."
        )
        save_notice = st.session_state.pop("manual_holdings_save_notice", None)
        if save_notice:
            st.success(str(save_notice))
        json_path = Path(data["manual_external_json"])
        current_json = json_path.read_text(encoding="utf-8") if json_path.exists() else '{\n  "positions": []\n}\n'
        with st.form("manual_external_holdings_editor"):
            edited_json = st.text_area(
                "Manual holdings JSON",
                value=current_json,
                height=560,
                key="manual_external_holdings_json_text",
                help="Edit the JSON directly. Nothing is written until Save is pressed and validation passes.",
            )
            save_column, note_column = st.columns([1, 3])
            save_requested = save_column.form_submit_button(
                "Save JSON & Refresh",
                type="primary",
                use_container_width=True,
            )
            note_column.caption("The previous valid file is retained as manual_external_holdings.json.bak.")
        if save_requested:
            try:
                parsed_payload = json.loads(edited_json)
                saved_rows, backup_path = write_manual_external_positions_file(parsed_payload, json_path)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                st.error(f"Manual holdings were not saved: {exc}")
            else:
                load_live_portfolio_data.clear()
                backup_note = f" Backup: {display_path(backup_path)}." if backup_path else ""
                st.session_state["manual_holdings_save_notice"] = (
                    f"Saved and synchronized {saved_rows} manual holding row(s).{backup_note}"
                )
                st.rerun()
