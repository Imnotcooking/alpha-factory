import streamlit as st

def load_css():
    custom_css = """
    <style>
        /* 1. Global Font and Seamless Background */
        html, body, [class*="st-"] {
            font-family: 'Inter', -apple-system, sans-serif !important;
        }

        .stApp {
            background-color: #0B0A10 !important;
            color: #E2E8F0 !important;
        }

        /* Remove the 'Box within a box' padding/background */
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        [data-testid="stMainBlockContainer"] {
            background-color: transparent !important;
            padding-top: 2rem !important;
        }

        /* 2. Sidebar Refinement */
        [data-testid="stSidebar"] {
            background-color: #12101A !important;
            border-right: 1px solid rgba(255, 255, 255, 0.03) !important;
        }
        /* Force sidebar text to look sleek */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #94A3B8 !important;
            font-weight: 500 !important;
            font-size: 0.95rem !important;
        }

        /* 3. Soft Cards (No More Harsh Lines) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid rgba(255, 255, 255, 0.04) !important; /* Barely visible */
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.015) !important; /* Faint glass */
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important; /* Soft shadow instead of border */
            padding: 15px !important;
        }

        /* Input fields */
        .stSelectbox div[data-baseweb="select"], .stTextInput input, .stNumberInput input {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            color: white !important;
            border-radius: 8px !important;
        }

        /* 4. Typography & Metrics */
        h1, h2, h3, h4, h5 {
            color: #FFFFFF !important;
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.0rem !important;
            font-weight: 700 !important;
            color: #F8FAFC !important;
        }

        [data-testid="stMetricLabel"] {
            color: #64748B !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        /* Deltas (Muted Neon) */
        [data-testid="stMetricDelta"] svg { display: none; }
        [data-testid="stMetricDelta"] > div { font-weight: 600 !important; }
        .st-emotion-cache-1jlnxa2 { color: #2DD4BF !important; } /* Soft Teal instead of hard green */
        .st-emotion-cache-1qg05tj { color: #F43F5E !important; } /* Soft Rose instead of hard red */

        /* 5. Buttons - Clean & Subtle Glow */
        .stButton > button {
            background: linear-gradient(90deg, #7C3AED 0%, #4F46E5 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        /* Secondary Button (PDF Export) */
        .stDownloadButton > button {
            background: transparent !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #94A3B8 !important;
            border-radius: 10px !important;
        }
        .stDownloadButton > button:hover {
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
