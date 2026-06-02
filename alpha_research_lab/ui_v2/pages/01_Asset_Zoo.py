import streamlit as st
import pandas as pd
import plotly.express as px
import os

from config import get_plotly_template, TEXT
from ui_state import init_global_ui_state, apply_global_style, render_global_controls_in_sidebar

st.set_page_config(page_title="Asset Zoo | Microstructure Profiler", layout="wide")

init_global_ui_state()
apply_global_style()
render_global_controls_in_sidebar()

lang = st.session_state.lang
t = TEXT[lang]

header_left, header_right = st.columns([0.82, 0.18])
with header_left:
    st.title(t["asset_zoo_title"])
with header_right:
    st.caption(f"Theme: {st.session_state.theme_mode}")

st.markdown(t["asset_zoo_subtitle"])

@st.cache_data
def load_profiles():
    # Adjust path based on where your streamlit app is run from
    file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data_engine', 'metadata', 'asset_volatility_profiles.csv')
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

df = load_profiles()

if df is None:
    st.error(t["asset_zoo_missing"])
else:
    # --- TOP ROW: High-Level Metrics ---
    st.markdown(t["asset_cluster_dist"])
    col1, col2, col3 = st.columns(3)
    tier_counts = df['trading_tier'].value_counts()
    
    col1.metric(t["asset_tier1"], tier_counts.get("Tier 1 (Trend / Breakout)", 0))
    col2.metric(t["asset_tier2"], tier_counts.get("Tier 2 (Mean-Reverting / Quiet)", 0))
    col3.metric(t["asset_tier3"], tier_counts.get("Tier 3 (Toxic / Illiquid)", 0))

    st.divider()

    # --- THE CROWN JEWEL: Volatility vs Efficiency Scatter ---
    st.markdown(t["asset_radar"])
    
    fig = px.scatter(
        df,
        x="avg_volatility",
        y="avg_efficiency",
        size="median_dollar_volume",
        color="trading_tier",
        hover_name="ticker",
        hover_data={"cluster_confidence": True, "avg_natr": True},
        color_discrete_map={
            "Tier 1 (Trend / Breakout)": "#00FF00", # Neon Green
            "Tier 2 (Mean-Reverting / Quiet)": "#FFA500", # Orange
            "Tier 3 (Toxic / Illiquid)": "#FF0000" # Red
        },
        title="Asset Personality Matrix (Bubble Size = Liquidity)",
        template=get_plotly_template(st.session_state.theme_mode)
    )
    
    fig.update_layout(xaxis_title="Annualized Volatility (Risk)", yaxis_title="Efficiency Ratio (Trend Cleanliness)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- THE TARGETING BOARD ---
    st.markdown(t["asset_target_board"])
    
    # Filter by Tier
    selected_tier = st.selectbox(t["asset_select_target"], df['trading_tier'].unique())
    target_df = df[df['trading_tier'] == selected_tier].sort_values(by='avg_efficiency', ascending=False)
    
    st.dataframe(
        target_df[['ticker', 'avg_volatility', 'avg_efficiency', 'avg_natr', 'median_dollar_volume', 'cluster_confidence']],
        use_container_width=True,
        hide_index=True
    )
    
    # Quick Copy-Paste String for Factor Scripts
    st.markdown(t["asset_copy_list"])
    ticker_list = target_df['ticker'].tolist()
    st.code(f"target_universe = {ticker_list}")