import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    from config import get_theme_css
except ImportError:
    import importlib.util
    from pathlib import Path

    _config_path = Path(__file__).resolve().parent / "config.py"
    _spec = importlib.util.spec_from_file_location("_ui_v2_config", _config_path)
    if _spec is None or _spec.loader is None:
        raise
    _ui_config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ui_config)
    get_theme_css = _ui_config.get_theme_css


def init_global_ui_state():
    if "lang" not in st.session_state:
        st.session_state.lang = "EN"
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "LIGHT"


def toggle_lang():
    st.session_state.lang = "ZH" if st.session_state.lang == "EN" else "EN"


def apply_global_style():
    st.markdown(get_theme_css(st.session_state.theme_mode), unsafe_allow_html=True)


def render_global_controls_in_sidebar():
    lang_label = "🇨🇳 切换至中文" if st.session_state.lang == "EN" else "🇬🇧 Switch to English"
    st.sidebar.button(
        lang_label,
        on_click=toggle_lang,
        key="global_sidebar_lang_toggle",
        width="stretch",
    )
