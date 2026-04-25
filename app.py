import streamlit as st

st.set_page_config(page_title="Alpha Factory Mobile", page_icon="🏦", layout="centered")

# The simplified, Read-Only Mobile Navigation
page_1 = st.Page("phases/1_executive_dashboard.py", title="Executive Dashboard", icon="📱")
page_2 = st.Page("phases/2_ai_logbook.py", title="AI Logbook & Ledger", icon="📖")

pg = st.navigation([page_1, page_2])
pg.run()