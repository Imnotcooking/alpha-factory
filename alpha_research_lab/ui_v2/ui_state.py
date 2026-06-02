import streamlit as st
from config import get_theme_css


def init_global_ui_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "EN"
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "LIGHT"


def toggle_lang():
    st.session_state.lang = "ZH" if st.session_state.lang == "EN" else "EN"


def toggle_theme():
    st.session_state.theme_mode = "DARK" if st.session_state.theme_mode == "LIGHT" else "LIGHT"


def apply_global_style():
    st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)


def render_global_controls_in_sidebar():
    lang_label = "🇨🇳 切换至中文" if st.session_state.lang == "EN" else "🇬🇧 Switch to English"
    theme_label = "🌙 Dark Mode" if st.session_state.theme_mode == "LIGHT" else "☀️ Light Mode"
    st.sidebar.button(
        lang_label,
        on_click=toggle_lang,
        use_container_width=True,
        key="global_sidebar_lang_toggle",
    )
    st.sidebar.button(
        theme_label,
        on_click=toggle_theme,
        use_container_width=True,
        key="global_sidebar_theme_toggle",
    )
