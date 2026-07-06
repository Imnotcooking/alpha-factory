import os
import sys

import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from config import TEXT
from data_manager import DataManager
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar
from executive_dashboard_logic import ExecutiveDashboardView

st.set_page_config(page_title="Strategy Comparison", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang

dm = DataManager()
view = ExecutiveDashboardView(dm)
view.render(lang=lang, theme_mode=st.session_state.theme_mode)
