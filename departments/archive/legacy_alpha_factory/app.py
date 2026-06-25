import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import sqlite3
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Alpha Factory | Executive Console", layout="wide", page_icon="🏦")

# --- CSS HACKS FOR INSTITUTIONAL FEEL ---
st.markdown("""
    <style>
    .big-font { font-size:40px !important; font-weight: bold; }
    .terminal-box { background-color: #0E1117; padding: 15px; border-radius: 5px; font-family: monospace; color: #00FF00; border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LIVE DATA INJECTION (The Real-Time Hook)
# ==========================================
DB_PATH = "portfolio_memory.db"

def fetch_live_portfolio_state():
    # 1. The Fail-Safe (Before the bell rings)
    default_state = {
        "current_regime": "Awaiting Market Open...",
        "regime_color": "info",
        "total_nav": 0.00,
        "daily_pnl": 0.00,
        "active_agent": "Standby",
        "target_beta": 0.0
    }
    
    if not os.path.exists(DB_PATH):
        return default_state
        
    try:
        # 2. Connect to the Live Trading Ledger
        conn = sqlite3.connect(DB_PATH)
        
        # Pull the absolute latest row of data written by the Cron Job
        query = """
            SELECT current_regime, regime_color, total_nav, daily_pnl, active_agent, target_beta 
            FROM daily_state 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 3. Inject the real money into the dashboard
        if not df.empty:
            return df.iloc[0].to_dict()
        return default_state
        
    except Exception as e:
        # If the table is still building, do not crash the UI
        return default_state

# Execute the hook
live_state = fetch_live_portfolio_state()

# Map the database values to the UI variables
current_regime = live_state["current_regime"]
regime_color = live_state["regime_color"]
total_nav = float(live_state["total_nav"])
daily_pnl = float(live_state["daily_pnl"])
active_agent = live_state["active_agent"]
target_beta = float(live_state["target_beta"])

# ==========================================
# 1. DYNAMIC REGIME BANNER
# ==========================================
if regime_color == "error":
    st.error(f"🚨 **ACTIVE REGIME:** {current_regime} | Risk Protocols Engaged.")
elif regime_color == "success":
    st.success(f"📈 **ACTIVE REGIME:** {current_regime} | Alpha Capture Optimal.")
else:
    st.info(f"📊 **ACTIVE REGIME:** {current_regime} | Standard Operations.")

# ==========================================
# 2. THE NAV CONSOLE (The Pulse)
# ==========================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Fund NAV", f"${total_nav:,.2f}", f"${daily_pnl:.2f}")
c2.metric("Target Beta", f"{target_beta}", "-0.6 vs SPY")
c3.metric("Capital Deployed", "64.2%")
c4.metric("Active Strategy", active_agent)

st.markdown("---")

# ==========================================
# 3. AI LOGBOOK (The Brain) & NAV CURVE
# ==========================================
col_chart, col_logs = st.columns([2, 1])

with col_chart:
    st.markdown("### 📉 Live NAV Trajectory")
    # Generate dummy intraday NAV line
    times = pd.date_range(start="09:30", periods=60, freq="1Min")
    nav_curve = total_nav + np.cumsum(np.random.normal(0, 5, 60))
    fig = go.Figure(go.Scatter(x=times, y=nav_curve, mode='lines', line=dict(color='#00E676', width=3)))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_logs:
    st.markdown("### 🧠 PPO Commander Logs")
    # Terminal style logging
    logs = f"""
    [{datetime.now().strftime('%H:%M:%S')}] SYS_BOOT: CRON Triggered.
    [{datetime.now().strftime('%H:%M:%S')}] VQ_ENCODER: State Space mapped.
    [{datetime.now().strftime('%H:%M:%S')}] CLUSTER: Archetype 2 Detected.
    [{datetime.now().strftime('%H:%M:%S')}] PPO_AGENT: Penalty matrix evaluated.
    [{datetime.now().strftime('%H:%M:%S')}] ACTION: Throttle Beta to 0.4.
    [{datetime.now().strftime('%H:%M:%S')}] ROUTER: Engaging {active_agent}.
    [{datetime.now().strftime('%H:%M:%S')}] OPTIMIZER: Convex weights solved.
    """
    st.markdown(f'<div class="terminal-box">{logs.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

# ==========================================
# 4. EXECUTION LEDGER (AC + MPC Monitor)
# ==========================================
st.markdown("---")
st.markdown("### ⚡ Execution Micro-Structure (AC + MPC)")

# Simulated Execution Data showcasing the new AC/MPC logic
exec_data = pd.DataFrame({
    "Ticker": ["TSLA", "TSLA", "AAPL", "NVDA"],
    "AC Trajectory": ["Slice 1/5", "Slice 2/5", "Slice 1/1", "Slice 1/3"],
    "MPC Order Type": ["LIMIT (Passive)", "MARKET (Aggressive)", "VWAP", "LIMIT (Passive)"],
    "Shares": [20, 15, 100, 50],
    "Fill Price": ["$185.42", "$185.50", "$170.10", "$890.05"],
    "Slippage (bps)": ["-0.2", "+1.1", "0.0", "-0.5"]
})

st.dataframe(exec_data, use_container_width=True, hide_index=True)