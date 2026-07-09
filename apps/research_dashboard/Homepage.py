import streamlit as st
import pandas as pd
import os
import sys
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
from universe_display import summarize_traded_universe, traded_universe_detail
from runtime_estimator import runtime_estimate_frame
from oqp.ui.asset_taxonomy import dashboard_taxonomy_frame

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

valid_run_ids = set(runs_df["run_id"].astype(str))
if st.session_state.selected_run is not None and str(st.session_state.selected_run) not in valid_run_ids:
    st.session_state.selected_run = None

# Build the Sidebar Buttons
for _, row in runs_df.iterrows():
    status_icon = "🔴" if pd.notna(row['failure_code']) else "🟢"
    
    # --- NEW: Safely extract Taxonomy & Universe ---
    asset_label = row.get('asset_class', 'FUTURES')
    tickers = summarize_traded_universe(
        row.get("traded_tickers", "ALL"),
        row.get("universe_size", 0),
        max_tickers=0,
    )
        
    # Inject the Asset Class and Universe into the button text
    label = f"{status_icon} {row['name']} (v{row['round_number']})\n[{asset_label} | {tickers}] | IC: {row['holdout_ic']:.4f}"
    
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

# --- NEW: MICROSTRUCTURE / ENVIRONMENT BADGE ---
asset_class = current_run.get('asset_class', 'N/A')
traded_tickers = current_run.get('traded_tickers', 'ALL')
universe_size = current_run.get('universe_size', 0)
universe_label = traded_universe_detail(traded_tickers, universe_size)

st.info(f"**🏛️ Asset Class:** `{asset_class}` &nbsp;&nbsp;|&nbsp;&nbsp; **🎯 Traded Universe:** `{universe_label}`")

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
    returns_path = current_run.get('returns_file_path', None)
    tearsheet_view.render(
        st.session_state.selected_run,
        current_run,
        returns_path=returns_path,
        lang=st.session_state.lang,
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
