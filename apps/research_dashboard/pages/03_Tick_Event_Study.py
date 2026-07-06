import os
import sys

import streamlit as st

UI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if UI_DIR not in sys.path:
    sys.path.insert(0, UI_DIR)

PROJECT_ROOT = os.path.dirname(UI_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if os.environ.get("OQP_EMBEDDED_STREAMLIT_PAGE") != "1":
    st.set_page_config(page_title="Tick Event Study", layout="wide")

from tick_pulse_lab.dashboard import TickPulseLabPage

TickPulseLabPage().render()
