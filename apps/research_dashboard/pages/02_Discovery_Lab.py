from __future__ import annotations

import runpy
import sys
from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from oqp.ui.translations import research_page_legacy_catalog
from ui_state import (
    apply_global_style,
    init_global_ui_state,
    render_global_controls_in_sidebar,
)


st.set_page_config(page_title="Discovery Lab", layout="wide")
init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

TEXT = research_page_legacy_catalog("discovery_lab")
lang = st.session_state.get("lang", "EN")
if lang == "CN":
    lang = "ZH"
copy = TEXT.get(lang, TEXT["EN"])

st.title(copy["title"])
st.caption(copy["subtitle"])

st.markdown(
    """
    <style>
    .st-key-discovery_lab_mode div[data-testid="stButtonGroup"] {
        width: 100% !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.28) !important;
    }
    .st-key-discovery_lab_mode div[role="radiogroup"] {
        width: 100% !important;
        gap: 1.5rem !important;
    }
    .st-key-discovery_lab_mode button[data-testid^="stBaseButton-segmented_control"] {
        min-height: 2.6rem;
        padding: 0.35rem 0 !important;
        border: 0 !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    .st-key-discovery_lab_mode button[data-testid="stBaseButton-segmented_controlActive"] {
        border-bottom-color: #ff4b4b !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
mode = st.segmented_control(
    copy["mode_label"],
    options=["pattern", "event_study", "relationships"],
    default="pattern",
    format_func=lambda value: copy[value],
    key="discovery_lab_mode",
    label_visibility="collapsed",
)

implementation = (
    APP_ROOT / "discovery" / "relationship_page.py"
    if mode == "relationships"
    else APP_ROOT / "discovery" / "pattern_page.py"
)

runpy.run_path(
    str(implementation),
    run_name="__main__",
    init_globals={
        "_OQP_EMBEDDED_STREAMLIT_PAGE": True,
        "_OQP_DISCOVERY_WORKFLOW_MODE": (
            "event_study" if mode == "event_study" else "discovery"
        ),
    },
)
