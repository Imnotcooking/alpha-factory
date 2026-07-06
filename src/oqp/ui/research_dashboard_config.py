# ==========================================
# GLOBAL SETTINGS
# ==========================================
from oqp.ui.translations import RESEARCH_TEXT


APP_TITLE = "Research Home"
APP_ICON = "📊"
LAYOUT = "wide"

# Custom CSS to shrink metric font sizes so they fit neatly in columns
CUSTOM_CSS = """
<style>
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; 
}
</style>
"""


def get_theme_css(theme_mode: str) -> str:
    """Shared UI theme CSS for all Streamlit pages."""
    if theme_mode == "DARK":
        return """
<style>
.stApp { background-color: #0E1117; color: #FAFAFA; }
[data-testid="stSidebar"] { background-color: #151A24; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #FAFAFA !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #FAFAFA !important; }
.stDataFrame, .stTable { color: #FAFAFA !important; }
</style>
"""
    return """
<style>
.stApp { background-color: #FFFFFF; color: #111827; }
[data-testid="stSidebar"] { background-color: #F8FAFC; }
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #111827 !important; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #111827 !important; }
.stDataFrame, .stTable { color: #111827 !important; }
</style>
"""


def get_plotly_template(theme_mode: str) -> str:
    return "plotly_dark" if theme_mode == "DARK" else "plotly_white"


# Compatibility alias for legacy research dashboard modules. The shared
# source of truth is now ``RESEARCH_TEXT`` in ``oqp.ui.translations``.
TEXT = {
    "EN": RESEARCH_TEXT["en"],
    "ZH": RESEARCH_TEXT["zh"],
}
