import streamlit as st
import pandas as pd
import os
import sys
from html import escape
from pathlib import Path

UI_DIR = os.path.dirname(os.path.abspath(__file__))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)
REPO_ROOT = Path(UI_DIR).parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 1. Core Architecture Imports
from config import APP_TITLE, APP_ICON, LAYOUT, CUSTOM_CSS, TEXT
from data_manager import DataManager
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar

# 2. View Modules Imports
from views.tearsheet_view import TearSheetView
from views.matrix_view import MatrixView
from views.dna_view import DNAView
from views.pareto_view import ParetoView
from views.ml_view import MLView
from views.assumptions_view import AssumptionsView
from universe_display import traded_universe_detail
from runtime_estimator import runtime_estimate_frame
from run_ledger import format_run_ledger_label
from oqp.ui.asset_taxonomy import dashboard_taxonomy_frame


def _context_label(copy: dict, key: str, fallback: str) -> str:
    return str(copy.get(key, fallback))


def _context_value(value, fallback: str = "unknown") -> str:
    if value is None or value is pd.NA:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else fallback


def _context_item(label: str, value: str) -> str:
    return (
        '<div class="oqp-context-item">'
        f'<div class="oqp-context-label">{escape(label)}</div>'
        f'<div class="oqp-context-value">{escape(value)}</div>'
        "</div>"
    )


def _render_run_context_card(asset_class: str, universe_label: str, scope: dict[str, object] | None, copy: dict) -> None:
    scope = scope or {}
    window = (
        f"{scope.get('start')} to {scope.get('end')}"
        if scope.get("start") and scope.get("end")
        else "unknown"
    )
    rows = int(scope.get("rows", 0) or 0)
    years = float(scope.get("years", 0.0) or 0.0)
    row_text = f"{rows:,} (~{years:.2f}y)" if rows else "unknown"
    source = _context_value(scope.get("source"))

    items = [
        (
            _context_label(copy, "context_run", "Run"),
            _context_value(scope.get("run_id")),
        ),
        (
            _context_label(copy, "context_window", "Result window"),
            window,
        ),
        (
            _context_label(copy, "context_rows", "Return rows"),
            row_text,
        ),
        (
            _context_label(copy, "context_prepared_data", "Prepared data"),
            _context_value(scope.get("prepared_window")),
        ),
        (
            _context_label(copy, "context_requested_filter", "Requested filter"),
            _context_value(scope.get("requested_window")),
        ),
        (
            _context_label(copy, "context_frequency", "Frequency"),
            _context_value(scope.get("frequency")),
        ),
        (
            _context_label(copy, "context_data", "Data"),
            _context_value(scope.get("role")),
        ),
        (
            _context_label(copy, "context_return_clock", "Return clock"),
            _context_value(scope.get("return_clock")),
        ),
        (
            _context_label(copy, "context_source", "Source"),
            source,
        ),
    ]
    item_html = "\n".join(_context_item(label, value) for label, value in items)
    style = """
<style>
.oqp-run-context {
    background: linear-gradient(135deg, #eaf4ff 0%, #edf7ff 55%, #f3fbff 100%);
    border: 1px solid #d8eafb;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1.05rem 1.15rem;
    margin: 0.75rem 0 1.15rem 0;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
}
.oqp-context-top {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem 1.2rem;
    align-items: center;
    padding-bottom: 0.85rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid rgba(59, 130, 246, 0.18);
}
.oqp-context-pill {
    display: inline-flex;
    align-items: baseline;
    gap: 0.4rem;
    color: #0f172a;
    font-size: 1.03rem;
    line-height: 1.25;
}
.oqp-context-pill strong {
    font-weight: 800;
}
.oqp-context-pill code {
    background: rgba(255, 255, 255, 0.62);
    border: 1px solid rgba(59, 130, 246, 0.16);
    border-radius: 5px;
    color: #0f172a;
    padding: 0.04rem 0.28rem;
    font-size: 0.92rem;
}
.oqp-context-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 0.65rem 1rem;
}
.oqp-context-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0;
    text-transform: uppercase;
    line-height: 1.2;
    margin-bottom: 0.18rem;
}
.oqp-context-value {
    color: #0f172a;
    font-size: 0.94rem;
    font-weight: 650;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
</style>
""".strip()
    asset_label = escape(_context_label(copy, "context_asset_class", "Asset Class"))
    asset_value = escape(_context_value(asset_class))
    universe_label_text = escape(_context_label(copy, "context_traded_universe", "Traded Universe"))
    universe_value = escape(_context_value(universe_label))
    card = (
        f"{style}\n"
        '<div class="oqp-run-context">'
        '<div class="oqp-context-top">'
        f'<span class="oqp-context-pill">🏛️ <strong>{asset_label}:</strong> <code>{asset_value}</code></span>'
        f'<span class="oqp-context-pill">🎯 <strong>{universe_label_text}:</strong> <code>{universe_value}</code></span>'
        "</div>"
        f'<div class="oqp-context-grid">{item_html}</div>'
        "</div>"
    )
    st.markdown(card, unsafe_allow_html=True)

# ==========================================
# PAGE CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title=APP_TITLE, layout=LAYOUT, page_icon=APP_ICON)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

init_global_ui_state()
if 'selected_run' not in st.session_state:
    st.session_state.selected_run = None
apply_global_style()

# Initialize Data Manager
# Avoid resource caching here so schema/query changes are reflected immediately.
dm = DataManager()

# Initialize Views
tearsheet_view = TearSheetView(dm)
matrix_view = MatrixView(dm)
dna_view = DNAView(dm)
pareto_view = ParetoView(dm)
ml_view = MLView(dm)
assumptions_view = AssumptionsView(dm)

# ==========================================
# SIDEBAR: THE LEDGER
# ==========================================
t = TEXT[st.session_state.lang]

render_global_controls_in_sidebar()
st.sidebar.markdown("---")

st.sidebar.markdown(t["ledger"])
st.sidebar.caption(t["ledger_desc"])

# Fetch all runs using the Data Manager
runs_df = dm.get_all_runs()

if runs_df.empty:
    st.warning(t.get("db_empty", "Database empty. Run the evaluator."))
    st.stop()

status_counts = runs_df["execution_status"].value_counts()
ledger_filter_options = {
    t.get(
        "ledger_completed",
        "✅ Completed backtests ({count})",
    ).format(count=int(status_counts.get("completed", 0))): "completed",
    t.get(
        "ledger_blocked",
        "⛔ Blocked before backtest ({count})",
    ).format(count=int(status_counts.get("blocked", 0))): "blocked",
    t.get(
        "ledger_missing",
        "⚠️ Missing return artifact ({count})",
    ).format(count=int(status_counts.get("artifact_missing", 0))): "artifact_missing",
    t.get(
        "ledger_all",
        "All ledger records ({count})",
    ).format(count=len(runs_df)): "all",
}
selected_filter_label = st.sidebar.selectbox(
    t.get("ledger_filter", "Ledger view"),
    options=list(ledger_filter_options),
    index=0,
    key="ledger_execution_filter",
)
selected_filter = ledger_filter_options[selected_filter_label]
ledger_runs_df = (
    runs_df
    if selected_filter == "all"
    else runs_df.loc[runs_df["execution_status"] == selected_filter]
)

valid_run_ids = set(ledger_runs_df["run_id"].astype(str))
if st.session_state.selected_run is not None and str(st.session_state.selected_run) not in valid_run_ids:
    st.session_state.selected_run = None

# Build the Sidebar Buttons
for _, row in ledger_runs_df.iterrows():
    label = format_run_ledger_label(row)
    if st.sidebar.button(label, key=row['run_id'], use_container_width=True):
        st.session_state.selected_run = row['run_id']

# ==========================================
# MAIN DASHBOARD (TOP SECTION)
# ==========================================
st.markdown(f"## {t['title']}")
with st.expander("Asset Taxonomy / 资产分类", expanded=False):
    st.caption(
        "Research runs, factor promotion, and future QMT execution use the same market lanes: "
        "US equities, US options, Chinese equities, Chinese options, and Chinese futures."
    )
    st.dataframe(
        dashboard_taxonomy_frame(language=st.session_state.lang.lower()),
        use_container_width=True,
        hide_index=True,
        height=230,
    )

# Top Metrics Panel
c1, c2, c3, c4 = st.columns(4)
best_ic = runs_df['holdout_ic'].max()
failure_rate = (runs_df['failure_code'].notna().sum() / len(runs_df)) * 100

c1.metric(t["metrics"][0], len(runs_df))
c2.metric(t["metrics"][1], f"{best_ic:.4f}")
c3.metric(t["metrics"][2], len(runs_df['name'].unique()))
c4.metric(t["metrics"][3], f"{failure_rate:.1f}%")

runtime_df = runtime_estimate_frame(runs_df)
if not runtime_df.empty:
    with st.expander("Runtime Estimates / 回测运行时间", expanded=False):
        st.caption(
            "Heuristic wall-clock ranges based on latest ledger row counts and asset route. "
            "The C++ execution pass is usually fast; factor computation often dominates."
        )
        st.dataframe(runtime_df, use_container_width=True, hide_index=True, height=220)

st.markdown("---")

# Render the Leaderboard via the MatrixView
matrix_view.render_leaderboard(lang=st.session_state.lang)

# ==========================================
# DETAILED FACTOR ANALYSIS (TABS)
# ==========================================
st.markdown("---")

if st.session_state.selected_run is None:
    st.info(t["select_run_hint"])
    st.stop()

# Get metadata for the specific run the user clicked
selected_run_rows = runs_df[runs_df["run_id"].astype(str) == str(st.session_state.selected_run)]
if selected_run_rows.empty:
    st.session_state.selected_run = None
    st.info(t["select_run_hint"])
    st.stop()
current_run = selected_run_rows.iloc[0]
execution_status = str(current_run.get("execution_status", "artifact_missing"))

if execution_status == "blocked":
    st.warning(
        t.get(
            "blocked_run_title",
            "⛔ Not backtested — this trial was blocked during preflight.",
        )
    )
    blocker_reason = _context_value(
        current_run.get("suggested_action"),
        t.get(
            "blocked_run_reason_missing",
            "No blocker explanation was recorded.",
        ),
    )
    st.error(
        t.get(
            "blocked_run_reason",
            "**Blocking reason:** {reason}",
        ).format(reason=blocker_reason)
    )
    st.caption(
        t.get(
            "blocked_run_explainer",
            "This is an audit record for a planned factor+sleeve trial, not a failed backtest. "
            "No return series should exist until the blocker is resolved.",
        )
    )
    assumptions_view.render(
        st.session_state.selected_run,
        current_run,
        lang=st.session_state.lang,
    )
    st.stop()

if execution_status == "artifact_missing":
    st.error(
        t.get(
            "missing_artifact_title",
            "⚠️ This ledger row has no loadable return artifact, so analytics cannot be shown.",
        )
    )
    st.stop()

# Selected-run context panel, rendered before the tab navbar.
asset_class = current_run.get('asset_class', 'N/A')
traded_tickers = current_run.get('traded_tickers', 'ALL')
universe_size = current_run.get('universe_size', 0)
universe_label = traded_universe_detail(traded_tickers, universe_size)
returns_path = current_run.get('returns_file_path', None)

try:
    context_returns = dm.get_run_returns(
        st.session_state.selected_run,
        returns_path=returns_path,
    )
except Exception:
    context_returns = pd.DataFrame()
context_manifest = tearsheet_view._load_manifest(st.session_state.selected_run)
context_scope = tearsheet_view.test_scope_summary(context_returns, current_run, context_manifest)
_render_run_context_card(asset_class, universe_label, context_scope, t)

tab_specs = [
    ("tearsheet", t["tab_tearsheet"]),
    ("dna", t["tab_dna"]),
    ("pareto", t["tab_pareto"]),
    ("correlation", t["tab_correlation"]),
]
if ml_view.should_render_tab(st.session_state.selected_run, current_run):
    tab_specs.append(("ml", t["tab_ml"]))
tab_specs.append(("assumptions", t.get("tab_assumptions", "Assumptions")))

tabs = dict(zip([key for key, _ in tab_specs], st.tabs([label for _, label in tab_specs])))

# Route the rendering logic to the designated OOP classes
with tabs["tearsheet"]:
    st.markdown(t["chart_title"])
    st.caption(t["chart_desc"])
    tearsheet_view.render(
        st.session_state.selected_run,
        current_run,
        returns_path=returns_path,
        lang=st.session_state.lang,
        show_test_scope=False,
    )

with tabs["dna"]:
    dna_view.render(
        st.session_state.selected_run,
        lang=st.session_state.lang,
        theme_mode=st.session_state.theme_mode,
    )

with tabs["pareto"]:
    pareto_view.render(current_run, lang=st.session_state.lang)

with tabs["correlation"]:
    matrix_view.render_correlation(lang=st.session_state.lang)

if "ml" in tabs:
    with tabs["ml"]:
        ml_view.render(
            st.session_state.selected_run,
            current_run,
            lang=st.session_state.lang,
            theme_mode=st.session_state.theme_mode,
        )

with tabs["assumptions"]:
    assumptions_view.render(
        st.session_state.selected_run,
        current_run,
        lang=st.session_state.lang,
    )
