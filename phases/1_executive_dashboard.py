import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import os
from dotenv import load_dotenv

st.markdown("## 📱 Alpha Factory NAV")

# --- ENVIRONMENT CHECK ---
load_dotenv()
TRADING_MODE = os.getenv("TRADING_MODE", "PAPER")

if TRADING_MODE == "LIVE":
    st.error("🔴 **LIVE TRADING ACTIVE**: This dashboard is tracking REAL capital.")
else:
    st.warning("🟠 **PAPER TRADING ACTIVE**: Simulated execution environment.")

# --- DATABASE CONNECTION ---
DB_PATH = "portfolio_memory.db"

def fetch_data(query):
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# --- SCOREBOARD ---
metrics_df = fetch_data("SELECT * FROM portfolio_metrics ORDER BY date DESC LIMIT 1")
available_cash = metrics_df['available_cash'].iloc[0] if not metrics_df.empty else 100000.00

positions_df = fetch_data("SELECT * FROM open_positions WHERE quantity > 0")

if positions_df.empty:
    total_invested = 0.0
else:
    positions_df['Invested_Value'] = positions_df['quantity'] * positions_df['average_price']
    total_invested = positions_df['Invested_Value'].sum()

total_nav = available_cash + total_invested

st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("Net Asset Value (NAV)", f"${total_nav:,.2f}")
col2.metric("Available Cash", f"${available_cash:,.2f}")

# --- INVENTORY CHART ---
st.markdown("### 📂 Live Allocations")
if positions_df.empty:
    st.info("Portfolio is 100% Cash.")
else:
    alloc_data = pd.DataFrame({
        'Asset': positions_df['ticker'].tolist() + ['CASH'],
        'Value': positions_df['Invested_Value'].tolist() + [available_cash]
    })
    
    fig = px.pie(
        alloc_data, values='Value', names='Asset', hole=0.7,
        color_discrete_sequence=px.colors.sequential.Tealgrn
    )
    fig.update_layout(
        template="plotly_dark", height=350, margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)