import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Core Architecture Imports
from config import APP_TITLE, APP_ICON, LAYOUT, CUSTOM_CSS, TEXT
from config import get_plotly_template
from data_manager import DataManager
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar, toggle_theme

# 2. View Modules Imports
from views.tearsheet_view import TearSheetView
from views.matrix_view import MatrixView
from views.dna_view import DNAView
from views.pareto_view import ParetoView
from executive_dashboard_logic import ExecutiveDashboardView

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
executive_view = ExecutiveDashboardView(dm)

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

# Build the Sidebar Buttons
for _, row in runs_df.iterrows():
    status_icon = "🔴" if pd.notna(row['failure_code']) else "🟢"
    
    # --- NEW: Safely extract Taxonomy & Universe ---
    asset_label = row.get('asset_class', 'FUTURES')
    tickers = str(row.get('traded_tickers', 'ALL'))
    if len(tickers) > 15:
        tickers = tickers[:12] + "..." # Truncate so the button doesn't overflow
        
    # Inject the Asset Class and Universe into the button text
    label = f"{status_icon} {row['name']} (v{row['round_number']})\n[{asset_label} | {tickers}] | IC: {row['holdout_ic']:.4f}"
    
    if st.sidebar.button(label, key=row['run_id'], use_container_width=True):
        st.session_state.selected_run = row['run_id']

# ==========================================
# MAIN DASHBOARD (TOP SECTION)
# ==========================================
top_left, top_right = st.columns([0.82, 0.18])
with top_left:
    st.markdown(f"## {t['title']}")
with top_right:
    theme_label = "🌙 Dark Mode" if st.session_state.theme_mode == "LIGHT" else "☀️ Light Mode"
    st.button(
        theme_label,
        on_click=toggle_theme,
        use_container_width=True,
        key="global_top_theme_toggle",
    )

# Top Metrics Panel
c1, c2, c3, c4 = st.columns(4)
best_ic = runs_df['holdout_ic'].max()
failure_rate = (runs_df['failure_code'].notna().sum() / len(runs_df)) * 100

c1.metric(t["metrics"][0], len(runs_df))
c2.metric(t["metrics"][1], f"{best_ic:.4f}")
c3.metric(t["metrics"][2], len(runs_df['name'].unique()))
c4.metric(t["metrics"][3], f"{failure_rate:.1f}%")

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
current_run = runs_df[runs_df['run_id'] == st.session_state.selected_run].iloc[0]

# --- NEW: MICROSTRUCTURE / ENVIRONMENT BADGE ---
asset_class = current_run.get('asset_class', 'N/A')
traded_tickers = current_run.get('traded_tickers', 'ALL')
universe_size = current_run.get('universe_size', 0)

st.info(f"**🏛️ Asset Class:** `{asset_class}` &nbsp;&nbsp;|&nbsp;&nbsp; **🎯 Traded Universe:** `{traded_tickers}` (Total Pool: {universe_size})")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t["tab_tearsheet"], 
    t["tab_correlation"], 
    t["tab_dna"], 
    t["tab_pareto"],
    t["tab_ml"],
    t["tab_conditional_perf"],
])

# Route the rendering logic to the designated OOP classes
with tab1:
    st.markdown(t["chart_title"])
    st.caption(t["chart_desc"])
    tearsheet_view.render(st.session_state.selected_run, current_run, lang=st.session_state.lang)

with tab2:
    matrix_view.render_correlation(lang=st.session_state.lang)

with tab3:
    dna_view.render(st.session_state.selected_run, lang=st.session_state.lang)

with tab4:
    pareto_view.render(current_run, lang=st.session_state.lang)

with tab5:
    st.markdown(f"### {t['tab_ml']}")
    st.caption(t["ml_caption"])
    
    # We kept Tab 5 inline here because it's only one simple chart and one database call
    df_importance = dm.get_feature_importance(st.session_state.selected_run)
    
    if not df_importance.empty:
        df_importance = df_importance.sort_values(by='importance', ascending=True)
        fig_imp = px.bar(
            df_importance, x='importance', y='feature', orientation='h',
            color='importance', color_continuous_scale='Viridis'
        )
        chart_template = get_plotly_template(st.session_state.theme_mode)
        fig_imp.update_layout(
            template=chart_template, height=600, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Relative Importance", yaxis_title=""
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info(t["ml_missing"])

with tab6:
    executive_view.render(lang=st.session_state.lang, theme_mode=st.session_state.theme_mode)