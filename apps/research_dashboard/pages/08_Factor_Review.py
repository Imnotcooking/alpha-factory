from __future__ import annotations

import os
import sys

import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

from ui_state import apply_global_style, init_global_ui_state, render_global_controls_in_sidebar
from views.factor_promotion_view import FactorPromotionView


st.set_page_config(page_title="Factor Review", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

view = FactorPromotionView()
view.render(lang=st.session_state.lang, theme_mode=st.session_state.theme_mode)
