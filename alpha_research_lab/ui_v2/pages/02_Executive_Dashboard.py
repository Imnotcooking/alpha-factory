import streamlit as st

from config import TEXT
from data_manager import DataManager
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar
from executive_dashboard_logic import ExecutiveDashboardView

st.set_page_config(page_title="Executive Dashboard", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang
t = TEXT[lang]

st.title(t["exec_page_title"])
st.caption(t["exec_page_subtitle"])

dm = DataManager()
view = ExecutiveDashboardView(dm)
view.render(lang=lang, theme_mode=st.session_state.theme_mode)
